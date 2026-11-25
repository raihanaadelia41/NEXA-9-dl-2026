"""Pembanding non-deep learning untuk klasifikasi 6 lintasan arah gerak.

Tugas tambahan Analis Utama (NIM 123450015).
Metode: Statistik sudut sendi dan kecepatan titik pose + Random Forest
Sesuai rancangan eksperimen Tim 9 (t09-arah-gerak.yaml & protokol.md).
Dilatih pada data latih dari protokol tim (src.protokol.data),
dievaluasi pada validasi (dan uji bila --final), ditulis ke
results/baseline/metrics.json lewat evaluate.

Jalankan:
    python -m src.pembanding_non_dl --seed 2026
    python -m src.pembanding_non_dl --seed 1
    python -m src.pembanding_non_dl --seed 1 --final
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np

from src.common import manifest

AKAR = manifest.AKAR


def ekstraksi_fitur_pose(X: np.ndarray) -> np.ndarray:
    """Ekstraksi statistik deskriptif, kecepatan, percepatan, dan sudut dari urutan pose."""
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.ndim == 2:
        return X

    N, T, F = X.shape
    fitur_list = []

    # 1. Statistik deskriptif sepanjang waktu
    fitur_list.append(np.mean(X, axis=1))          # (N, F)
    fitur_list.append(np.std(X, axis=1))           # (N, F)
    fitur_list.append(np.min(X, axis=1))           # (N, F)
    fitur_list.append(np.max(X, axis=1))           # (N, F)
    fitur_list.append(np.median(X, axis=1))        # (N, F)
    fitur_list.append(X[:, -1, :] - X[:, 0, :])    # (N, F) pergeseran neto / arah gerak

    # 2. Kecepatan titik pose (perbedaan temporal orde 1)
    if T > 1:
        v = np.diff(X, axis=1)                     # (N, T-1, F)
        fitur_list.append(np.mean(v, axis=1))
        fitur_list.append(np.std(v, axis=1))
        fitur_list.append(np.max(v, axis=1))
        fitur_list.append(np.min(v, axis=1))

        # 3. Percepatan titik pose (perbedaan temporal orde 2)
        if T > 2:
            a = np.diff(v, axis=1)                 # (N, T-2, F)
            fitur_list.append(np.mean(a, axis=1))
            fitur_list.append(np.std(a, axis=1))

    # 4. Statistik sudut (pasangan fitur berdekatan)
    sudut_list = []
    for j in range(0, F - 1, 2):
        sudut = np.arctan2(X[:, :, j + 1], X[:, :, j] + 1e-7)
        sudut_list.append(np.mean(sudut, axis=1, keepdims=True))
        sudut_list.append(np.std(sudut, axis=1, keepdims=True))
    if sudut_list:
        fitur_list.append(np.hstack(sudut_list))

    # 5. Nilai sekuens mentah terentang (flattened)
    fitur_list.append(X.reshape(N, -1))

    return np.hstack(fitur_list)


def _standar(Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    Xtr_std = (Xtr - mu) / sd
    Xva_std = (Xva - mu) / sd
    Xte_std = (Xte - mu) / sd if Xte is not None else None
    return Xtr_std, Xva_std, Xte_std


def latih_random_forest(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, seed: int) -> tuple[np.ndarray, Any]:
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        random_state=seed,
        class_weight="balanced",
    )
    rf.fit(Xtr, ytr)
    pred_val = rf.predict(Xva)
    return pred_val, rf


def logistik(Xtr, ytr, Xva, seed: int, epoch: int = 20, lr: float = 0.1, batch: int = 32) -> np.ndarray:
    """Fallback regresi logistik softmax NumPy bila sklearn tidak tersedia."""
    rng = np.random.default_rng(seed)
    k = int(ytr.max()) + 1
    W = np.zeros((Xtr.shape[1], k), np.float32); b = np.zeros(k, np.float32)
    n = len(Xtr)
    for _ in range(epoch):
        for i in rng.permutation(n).reshape(-1)[: n - n % batch].reshape(-1, batch):
            z = Xtr[i] @ W + b
            z -= z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            p[np.arange(batch), ytr[i]] -= 1
            p /= batch
            W -= lr * (Xtr[i].T @ p + 1e-4 * W); b -= lr * p.sum(0)
    return (Xva @ W + b).argmax(1)


def main() -> None:
    from src.protokol import data

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--metode", default="rf", choices=["rf", "logistik"],
                    help="metode pembanding non-DL: rf (Random Forest) | logistik")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    # Ekstraksi fitur pose + kecepatan + sudut
    Xtr_feat = ekstraksi_fitur_pose(Xtr)
    Xva_feat = ekstraksi_fitur_pose(Xva)
    Xte_feat = ekstraksi_fitur_pose(Xte) if (a.final and Xte is not None) else None

    Xtr_std, Xva_std, Xte_std = _standar(Xtr_feat, Xva_feat, Xte_feat)

    pred_test: np.ndarray | None = None
    ytr_arr = np.asarray(ytr, dtype=np.int64)
    if a.metode == "rf":
        try:
            pred_val, model_rf = latih_random_forest(Xtr_std, ytr_arr, Xva_std, a.seed)
            if Xte_std is not None:
                pred_test = model_rf.predict(Xte_std).astype(np.int64)
        except ImportError:
            pred_val = logistik(Xtr_std, ytr_arr, Xva_std, a.seed).astype(np.int64)
    else:
        pred_val = logistik(Xtr_std, ytr_arr, Xva_std, a.seed).astype(np.int64)

    durasi = time.time() - t0
    out = AKAR / "results" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val, dtype=np.int64))

    cmd = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", "baseline",
        "--model", "baseline",
        "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat,
    ]

    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test, dtype=np.int64))
        cmd += ["--pred-test", str(out / "pred_test.npy")]

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run(cmd, check=True, cwd=AKAR, env=env)


if __name__ == "__main__":
    main()
