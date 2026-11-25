import os
import numpy as np
import pandas as pd

SEQ_LEN = 70  # 7 detik setelah buang 1 detik awal (@ 10 fps)

def _resample_window(arr, target_len=SEQ_LEN):
    if len(arr) == 0:
        return np.zeros((target_len, arr.shape[1]), dtype=np.float32)
    if len(arr) == target_len:
        return arr.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.stack([np.interp(x_new, x_old, arr[:, j]) for j in range(arr.shape[1])], axis=1)
    return out.astype(np.float32)

def _normalisasi_fit_transform(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std

def _data_simulasi(seq_len=SEQ_LEN, n_features=34, seed=1009):
    print("[FALLBACK-SIMULASI] Berkas riil tidak lengkap. Menggunakan data simulasi (108 sampel).")
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(108, seq_len, n_features)).astype(np.float32)
    y = np.tile(np.arange(6), 18).astype(np.int64)
    # 72 train (P01-P04), 18 val (P05), 18 test (P06)
    X_train, y_train = X[:72], y[:72]
    X_val, y_val = X[72:90], y[72:90]
    X_test, y_test = X[90:], y[90:]
    X_train, X_val, X_test = _normalisasi_fit_transform(X_train, X_val, X_test)
    return (X_train, X_val, X_test), (y_train, y_val, y_test)

def muat(sesi_dir="data/sesi_tim09", seq_len=SEQ_LEN):
    label_path = os.path.join(sesi_dir, "label.csv")
    pose_path = os.path.join(sesi_dir, "pose.csv")
    digital_path = os.path.join(sesi_dir, "digital.csv")
    imu_path = os.path.join(sesi_dir, "imu.csv")

    if not os.path.exists(label_path):
        return _data_simulasi(seq_len=seq_len)

    df_lbl = pd.read_csv(label_path)

    # Pilih sumber fitur: Prioritas 1 = pose.csv, Prioritas 2 = imu.csv, Prioritas 3 = simulasi
    if os.path.exists(pose_path) and os.path.getsize(pose_path) > 50:
        print("[DATA] Menggunakan fitur utama: pose.csv")
        df_feat = pd.read_csv(pose_path).sort_values("t_host")
        # Gabung digital.csv HANYA jika ada isinya (tidak kosong)
        if os.path.exists(digital_path) and os.path.getsize(digital_path) > 50:
            df_dig = pd.read_csv(digital_path)
            if len(df_dig) > 0:
                df_dig = df_dig.sort_values("t_host")
                df_feat = pd.merge_asof(df_feat, df_dig, on="t_host", direction="nearest")
            else:
                print("[INFO] digital.csv kosong (0 baris), dilewati.")
    elif os.path.exists(imu_path) and os.path.getsize(imu_path) > 50:
        print("[FALLBACK-IMU] pose.csv belum ada. Melatih sementara menggunakan fitur riil imu.csv.")
        df_feat = pd.read_csv(imu_path, usecols=["t_host", "ax", "ay", "az", "gx", "gy", "gz"]).sort_values("t_host")
    else:
        return _data_simulasi(seq_len=seq_len)

    fitur_cols = [c for c in df_feat.select_dtypes(include=[np.number]).columns if c not in ("t_host", "t_esp_ms")]
    df_feat[fitur_cols] = df_feat[fitur_cols].fillna(0.0)

    kelas_unik = sorted(df_lbl["lintasan"].astype(str).unique())
    label_to_id = {k: i for i, k in enumerate(kelas_unik)}

    X_list, y_list, p_list = [], [], []
    for _, row in df_lbl.iterrows():
        t0 = float(row["t_mulai_host"]) + 1.0  # Protokol: buang 1 detik pertama
        t1 = float(row["t_selesai_host"])
        seg = df_feat[(df_feat["t_host"] >= t0) & (df_feat["t_host"] <= t1)][fitur_cols].to_numpy()
        if len(seg) < 2:
            continue
        X_list.append(_resample_window(seg, target_len=seq_len))
        y_list.append(label_to_id[str(row["lintasan"])])
        p_list.append(str(row["partisipan"]))

    if len(X_list) == 0:
        print("[PERINGATAN] Tidak ada irisan timestamp yang cocok antara label.csv dan fitur.")
        return _data_simulasi(seq_len=seq_len)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    parts = np.array(p_list)

    # Split per partisipan: P01-P04 train, P05 val, P06 test
    m_val = parts == "P05"
    m_test = parts == "P06"
    m_train = ~(m_val | m_test)

    assert set(parts[m_train]).isdisjoint(set(parts[m_val])), "Leakage train-val!"
    assert set(parts[m_train]).isdisjoint(set(parts[m_test])), "Leakage train-test!"

    X_train, y_train = X[m_train], y[m_train]
    X_val, y_val = X[m_val], y[m_val]
    X_test, y_test = X[m_test], y[m_test]

    X_train, X_val, X_test = _normalisasi_fit_transform(X_train, X_val, X_test)
    print(f"[OK] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape} | Kelas: {kelas_unik}")
    return (X_train, X_val, X_test), (y_train, y_val, y_test)

"""Pemuat data kasus Tim 9 — pose, PIR, dan sensor pintu."""
'''
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import manifest

TUGAS = "klasifikasi"

LABEL_MAP = {
    "masuk dari kiri": 0,
    "keluar ke kiri": 1,
    "masuk dari kanan": 2,
    "keluar ke kanan": 3,
    "melintas tanpa berhenti": 4,
    "berhenti di meja": 5,
}


def _normalisasi_fit_transform(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalisasi z-score dihitung hanya dari data latih."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def muat(
    sesi_dir: str | Path = "data/sesi_tim09",
    seq_len: int = 70,
    train_p: tuple[str, ...] = ("P01", "P02", "P03", "P04"),
    val_p: tuple[str, ...] = ("P05",),
    test_p: tuple[str, ...] = ("P06",),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Muat dan potong jendela data sesi Tim 9 (pose + PIR + sensor pintu).

    Buang satu detik pertama setiap kondisi, pisahkan per partisipan, lalu
    normalisasi menggunakan statistik data latih saja. Nilai balik mengikuti
    kontrak protokol: X_train, y_train, X_val, y_val, X_test, y_test.
    """
    if seq_len <= 0:
        raise ValueError("seq_len harus lebih besar dari 0")

    train_p, val_p, test_p = set(train_p), set(val_p), set(test_p)
    if not train_p.isdisjoint(val_p):
        raise ValueError("Kebocoran partisipan antara latih dan validasi")
    if not train_p.isdisjoint(test_p):
        raise ValueError("Kebocoran partisipan antara latih dan uji")
    if not val_p.isdisjoint(test_p):
        raise ValueError("Kebocoran partisipan antara validasi dan uji")

    sesi_path = Path(sesi_dir)
    if not sesi_path.is_absolute():
        sesi_path = manifest.AKAR / sesi_path
    label_path = sesi_path / "label.csv"
    digital_path = sesi_path / "digital.csv"
    pose_path = sesi_path / "pose.csv"

    if not label_path.exists() or not pose_path.exists():
        print(
            f"[INFO] Berkas label.csv atau pose.csv belum ditemukan di {sesi_path}; "
            "menggunakan data simulasi untuk uji kerangka."
        )
        rng = np.random.default_rng(1009)
        feat_dim = 34  # 32 fitur pose + PIR + sensor pintu
        X_train = rng.normal(size=(72, seq_len, feat_dim)).astype(np.float32)
        y_train = rng.integers(0, len(LABEL_MAP), size=72, dtype=np.int64)
        X_val = rng.normal(size=(18, seq_len, feat_dim)).astype(np.float32)
        y_val = rng.integers(0, len(LABEL_MAP), size=18, dtype=np.int64)
        X_test = rng.normal(size=(18, seq_len, feat_dim)).astype(np.float32)
        y_test = rng.integers(0, len(LABEL_MAP), size=18, dtype=np.int64)
        X_train, X_val, X_test = _normalisasi_fit_transform(X_train, X_val, X_test)
        return X_train, y_train, X_val, y_val, X_test, y_test

    df_label = pd.read_csv(label_path)
    df_pose = pd.read_csv(pose_path).sort_values("t_host")
    df_dig = pd.read_csv(digital_path).sort_values("t_host") if digital_path.exists() else None

    if df_dig is not None and not df_dig.empty:
        dig_cols = [col for col in ("t_host", "pir", "pintu") if col in df_dig.columns]
        df_feat = pd.merge_asof(df_pose, df_dig[dig_cols], on="t_host", direction="backward")
        for col in ("pir", "pintu"):
            if col not in df_feat.columns:
                df_feat[col] = 0.0
            df_feat[col] = df_feat[col].fillna(0.0)
    else:
        df_feat = df_pose.copy()
        df_feat["pir"] = 0.0
        df_feat["pintu"] = 0.0

    feature_cols = [col for col in df_feat.columns if col != "t_host"]
    splits: dict[str, tuple[list[np.ndarray], list[int]]] = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }

    for _, cond in df_label.iterrows():
        participant = str(cond["partisipan"])
        label = LABEL_MAP.get(str(cond["lintasan"]).strip().lower())
        if label is None:
            continue

        t_start = float(cond.get("t_mulai", cond.get("t_start", 0.0))) + 1.0
        t_end = float(cond.get("t_selesai", cond.get("t_end", t_start + 7.0)))
        segment = df_feat.loc[
            (df_feat["t_host"] >= t_start) & (df_feat["t_host"] <= t_end),
            feature_cols,
        ].to_numpy(dtype=np.float32)
        if len(segment) == 0:
            continue

        if len(segment) >= seq_len:
            window = segment[:seq_len]
        else:
            padding = np.zeros((seq_len - len(segment), segment.shape[1]), dtype=np.float32)
            window = np.vstack((segment, padding))

        destination = (
            "train" if participant in train_p
            else "val" if participant in val_p
            else "test" if participant in test_p
            else None
        )
        if destination is not None:
            splits[destination][0].append(window)
            splits[destination][1].append(label)

    empty_splits = [name for name, (features, _) in splits.items() if not features]
    if empty_splits:
        raise ValueError(
            "Tidak ada sampel untuk split: "
            + ", ".join(empty_splits)
            + ". Periksa label.csv dan daftar partisipan."
        )

    X_train, y_train = np.asarray(splits["train"][0], dtype=np.float32), np.asarray(splits["train"][1], dtype=np.int64)
    X_val, y_val = np.asarray(splits["val"][0], dtype=np.float32), np.asarray(splits["val"][1], dtype=np.int64)
    X_test, y_test = np.asarray(splits["test"][0], dtype=np.float32), np.asarray(splits["test"][1], dtype=np.int64)
    X_train, X_val, X_test = _normalisasi_fit_transform(X_train, X_val, X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test
'''