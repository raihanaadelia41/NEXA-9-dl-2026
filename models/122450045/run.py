"""Model Bidirectional GRU (jalur RNN) untuk klasifikasi 6 lintasan arah gerak.

Identitas:
    NIM  : 122450045
    Nama : Eka Fidiya Putri
    Jalur: rnn

Jalankan:
    python models/122450045/run.py --seed 1
    python models/122450045/run.py --seed 1 --final    # sprint terakhir, sekali
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name  # otomatis: 122450045
JALUR = "rnn"

SEQ_LEN = 70
JUMLAH_KELAS = 6
BATCH_SIZE = 8
MAX_EPOCHS = 30
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.2
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0


def _pilih_device(perangkat: str) -> tuple["torch.device", str]:
    """Menentukan torch.device dan label perangkat untuk src.common.evaluate."""
    import torch

    nilai = perangkat.strip()
    low = nilai.lower()

    if low == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu laptop"

    if low.startswith("cpu"):
        label = "cpu laptop" if low == "cpu" else nilai
        return torch.device("cpu"), label

    if low.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diminta, tetapi tidak tersedia pada perangkat ini.")
        return torch.device(low), nilai

    raise ValueError(
        f"Perangkat tidak dikenal: {perangkat!r}. "
        "Gunakan 'auto', 'cpu', 'cpu laptop', atau 'cuda'."
    )


def _validasi_dan_tensor_fitur(
    X: np.ndarray,
    expected_features: int | None = None,
) -> "torch.Tensor":
    """Validasi dimensi, ketiadaan NaN/Inf, lalu konversi ke float32 Tensor (N, T, F)."""
    import torch

    arr = np.asarray(X, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(
            "Input GRU harus berbentuk 3D (sampel, waktu, fitur). "
            f"Diperoleh: {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("Dataset input tidak boleh kosong.")
    if arr.shape[1] != SEQ_LEN:
        raise ValueError(
            f"Panjang sekuens harus {SEQ_LEN}, diperoleh {arr.shape[1]}."
        )
    if expected_features is not None and arr.shape[2] != expected_features:
        raise ValueError(
            f"Jumlah fitur harus {expected_features}, diperoleh {arr.shape[2]}."
        )
    if not np.isfinite(arr).all():
        raise ValueError("Input mengandung nilai NaN atau Infinity.")

    return torch.from_numpy(np.ascontiguousarray(arr))


def _validasi_dan_tensor_label(y: np.ndarray, n_samples: int, nama: str) -> "torch.Tensor":
    """Validasi label 1D tidak kosong, finite, bilangan bulat, dan berada pada rentang 0..5."""
    import torch

    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"Label {nama} harus 1D, diperoleh shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"Label {nama} tidak boleh kosong.")
    if len(arr) != n_samples:
        raise ValueError(
            f"Jumlah sampel label {nama} ({len(arr)}) tidak sama dengan fitur ({n_samples})."
        )
    if np.issubdtype(arr.dtype, np.bool_) or not (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)
    ):
        raise ValueError(
            f"Label {nama} harus bertipe numerik bilangan bulat, diperoleh dtype {arr.dtype}."
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"Label {nama} mengandung nilai NaN atau Infinity.")
    if np.issubdtype(arr.dtype, np.floating) and not np.all(arr == np.floor(arr)):
        raise ValueError(
            f"Label {nama} bertipe float dengan nilai bukan bilangan bulat."
        )

    arr_int = arr.astype(np.int64)
    if arr_int.min() < 0 or arr_int.max() >= JUMLAH_KELAS:
        raise ValueError(
            f"Label {nama} harus berada pada rentang 0..{JUMLAH_KELAS - 1}, "
            f"diperoleh [{arr_int.min()}, {arr_int.max()}]."
        )
    return torch.from_numpy(np.ascontiguousarray(arr_int))


def _buat_model(
    input_dim: int,
    num_classes: int,
    pooling: str = "mean_hidden",
) -> "torch.nn.Module":
    """Membangun BiGRU dengan pilihan mean atau mean+hidden temporal pooling."""
    import torch
    import torch.nn as nn

    if pooling not in {"mean", "mean_hidden"}:
        raise ValueError(
            f"Pooling tidak dikenal: {pooling!r}. Gunakan 'mean' atau 'mean_hidden'."
        )

    class BiGRUClassifier(nn.Module):
        def __init__(
            self,
            feat_dim: int,
            hidden_size: int = HIDDEN_SIZE,
            num_layers: int = NUM_LAYERS,
            dropout: float = DROPOUT,
            n_classes: int = num_classes,
            pooling_mode: str = pooling,
        ) -> None:
            super().__init__()
            self.pooling = pooling_mode
            self.input_norm = nn.LayerNorm(feat_dim)
            self.gru = nn.GRU(
                input_size=feat_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )

            # Mean: 2H; mean_hidden: mean (2H) + dua hidden state (H + H).
            repr_dim = hidden_size * (2 if pooling_mode == "mean" else 4)
            self.head = nn.Sequential(
                nn.LayerNorm(repr_dim),
                nn.Dropout(p=dropout),
                nn.Linear(repr_dim, n_classes),
            )

        def forward(self, x):
            # x: (B, SEQ_LEN, input_dim)
            h = self.input_norm(x)
            out, h_n = self.gru(h)  # out: (B, T, 2H); h_n: (2, B, H)
            mean_pool = out.mean(dim=1)

            if self.pooling == "mean":
                pooled = mean_pool
            else:
                forward_final = h_n[-2]
                backward_final = h_n[-1]
                pooled = torch.cat(
                    [mean_pool, forward_final, backward_final],
                    dim=1,
                )

            return self.head(pooled)  # (B, num_classes)

    return BiGRUClassifier(
        feat_dim=input_dim,
        n_classes=num_classes,
        pooling_mode=pooling,
    )


def latih_dan_prediksi(
    seed: int,
    final: bool,
    device: "torch.device",
    pooling: str,
    simulasi: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Melatih BiGRU pada split latih dan menghasilkan prediksi validasi (serta uji bila final)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # 1. Seed reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Muat data resmi dari protokol tim
    if simulasi:
        print("[MODE] Menggunakan data simulasi.")
        Xtr, ytr, Xva, yva, Xte, _ = data.muat(
            sesi_dir="data/__paksa_mode_simulasi__"
        )
    else:
        Xtr, ytr, Xva, yva, Xte, _ = data.muat()

    X_train = _validasi_dan_tensor_fitur(Xtr)
    input_dim = X_train.shape[-1]
    y_train = _validasi_dan_tensor_label(ytr, len(X_train), "train")

    X_val = _validasi_dan_tensor_fitur(Xva, expected_features=input_dim)
    y_val = _validasi_dan_tensor_label(yva, len(X_val), "val")

    print(
        f"[RNN-BiGRU] pooling={pooling}, perangkat={device}, seed={seed}, "
        f"X_train={tuple(X_train.shape)}, X_val={tuple(X_val.shape)}, input_dim={input_dim}"
    )

    # 3. DataLoader dengan generator ber-seed
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    # 4. Model, loss, dan optimizer AdamW (dengan weight_decay)
    model = _buat_model(input_dim=input_dim, num_classes=JUMLAH_KELAS, pooling=pooling).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    X_val_dev = X_val.to(device)

    # y_val divalidasi untuk memastikan format label benar; tidak dipakai
    # untuk memilih bobot atau menghentikan training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
    )

    # Training dengan jumlah epoch tetap, scheduler, dan gradient clipping.
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            total_train_loss += loss.item() * len(xb)
            total_train_samples += len(xb)

        avg_train_loss = total_train_loss / total_train_samples
        scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == MAX_EPOCHS:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:2d}/{MAX_EPOCHS} | "
                f"train_loss={avg_train_loss:.4f} | lr={current_lr:.6g}"
            )

    model.eval()

    with torch.no_grad():
        pred_val = model(X_val_dev).argmax(dim=1).cpu().numpy().astype(np.int64).ravel()

        pred_test: np.ndarray | None = None
        if final:
            if Xte is None:
                raise ValueError("Flag --final diberikan, tetapi X_test dari data.muat() bernilai None.")
            X_test = _validasi_dan_tensor_fitur(Xte, expected_features=input_dim).to(device)
            pred_test = model(X_test).argmax(dim=1).cpu().numpy().astype(np.int64).ravel()

    return pred_val, pred_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model Bidirectional GRU (jalur RNN) untuk Capstone NEXA-9-dl-2026."
    )
    parser.add_argument("--seed", type=int, required=True, help="Seed pengacakan (mis. 1, 2, 3)")
    parser.add_argument("--final", action="store_true", help="Evaluasi data uji sekali pada sprint terakhir")
    parser.add_argument("--simulasi", action="store_true", help="Gunakan data simulasi untuk uji kerangka")
    parser.add_argument(
        "--pooling",
        choices=["mean", "mean_hidden"],
        default="mean_hidden",
        help="Strategi temporal pooling: mean atau mean_hidden",
    )
    parser.add_argument(
        "--perangkat",
        default="auto",
        help="Perangkat komputasi: 'auto', 'cpu', 'cpu laptop', atau 'cuda'",
    )
    args = parser.parse_args()

    device, label_perangkat = _pilih_device(args.perangkat)

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(
        seed=args.seed,
        final=args.final,
        device=device,
        pooling=args.pooling,
        simulasi=args.simulasi,
    )
    durasi = time.time() - t0

    if durasi > 1200.0:
        print(f"[PERINGATAN] Durasi ({durasi:.1f}s) melebihi batas anggaran 20 menit (1200s).")

    out_dir = AKAR / "results" / NIM
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.simulasi:
        path_pred_val = out_dir / f"pred_val_simulasi_{args.pooling}_seed{args.seed}.npy"
        np.save(path_pred_val, np.asarray(pred_val, dtype=np.int64))
        print("[SIMULASI] Training kerangka selesai.")
        print("[SIMULASI] Evaluator resmi dilewati; tidak membuat metrics.json.")
        print(f"[SIMULASI] Prediksi tersimpan: {path_pred_val.relative_to(AKAR)}")
        return

    path_pred_val = out_dir / f"pred_val_{args.pooling}.npy"
    np.save(path_pred_val, np.asarray(pred_val, dtype=np.int64))

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(args.seed),
        "--pred-val",
        str(path_pred_val),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        label_perangkat,
    ]

    if pred_test is not None:
        path_pred_test = out_dir / "pred_test.npy"
        np.save(path_pred_test, np.asarray(pred_test, dtype=np.int64))
        perintah += ["--pred-test", str(path_pred_test)]

    path_metrics = out_dir / "metrics.json"
    previous_metrics = path_metrics.read_bytes() if path_metrics.exists() else None

    try:
        subprocess.run(perintah, check=True, cwd=AKAR)
        if path_metrics.exists():
            path_metrics_pooling = out_dir / f"metrics_{args.pooling}.json"
            path_metrics_pooling.write_bytes(path_metrics.read_bytes())
    finally:
        if previous_metrics is not None:
            path_metrics.write_bytes(previous_metrics)
        elif path_metrics.exists():
            path_metrics.unlink()


if __name__ == "__main__":
    main()
