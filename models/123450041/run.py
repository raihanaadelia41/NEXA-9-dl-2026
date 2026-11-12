
"""
Model CNN 1D untuk Capstone Deep Learning NEXA.

Jalankan:
    python models/123450041/run.py --seed 1

Opsi --final hanya digunakan pada evaluasi akhir
sesuai protokol.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# KONFIGURASI
# ============================================================

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
JALUR = "cnn"

JUMLAH_KELAS = 6
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001


# ============================================================
# FUNGSI PREPROCESSING TENSOR
# ============================================================

def ke_tensor(X, torch):
    """
    Mengubah input dari bentuk:
        (sampel, waktu, fitur)

    Menjadi:
        (sampel, fitur, waktu)

    Format tersebut diperlukan oleh Conv1D PyTorch.
    """

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(
            "Input CNN 1D harus berbentuk "
            "(sampel, waktu, fitur). "
            f"Diperoleh: {X.shape}"
        )

    if X.shape[0] == 0:
        raise ValueError("Dataset tidak boleh kosong.")

    if not np.isfinite(X).all():
        raise ValueError(
            "Input mengandung nilai NaN atau Inf."
        )

    return torch.tensor(X).permute(
        0, 2, 1
    ).contiguous()


# ============================================================
# ARSITEKTUR CNN 1D
# ============================================================

def buat_model(jumlah_fitur, jumlah_kelas, nn):
    """
    Arsitektur CNN 1D:

    Conv1D -> ReLU -> MaxPooling
    -> Conv1D -> ReLU -> AdaptiveAvgPooling
    -> Flatten -> Fully Connected
    """

    model = nn.Sequential(

        # Convolutional block pertama
        nn.Conv1d(
            in_channels=jumlah_fitur,
            out_channels=32,
            kernel_size=3,
            padding=1
        ),

        nn.ReLU(),

        nn.MaxPool1d(
            kernel_size=2
        ),

        # Convolutional block kedua
        nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        ),

        nn.ReLU(),

        # Menghasilkan panjang fitur temporal 1
        nn.AdaptiveAvgPool1d(1),

        # Classification layer
        nn.Flatten(),

        nn.Linear(
            in_features=64,
            out_features=jumlah_kelas
        )
    )

    return model


# ============================================================
# TRAINING DAN PREDIKSI
# ============================================================

def latih_dan_prediksi(
    seed: int,
    final: bool,
    perangkat: str
) -> tuple[np.ndarray, np.ndarray | None]:

    import torch
    import torch.nn as nn

    from torch.utils.data import (
        DataLoader,
        TensorDataset
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if perangkat == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(perangkat)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA tidak tersedia pada perangkat ini."
        )

    print(f"Perangkat komputasi: {device}")

    # --------------------------------------------------------
    # Memuat dataset
    # --------------------------------------------------------

    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    X_train = ke_tensor(Xtr, torch).to(device)
    X_val = ke_tensor(Xva, torch).to(device)

    y_train = torch.tensor(
        np.asarray(ytr),
        dtype=torch.long
    )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Jumlah data training dan label tidak sama."
        )

    if y_train.ndim != 1:
        raise ValueError(
            "Label harus berbentuk array satu dimensi."
        )

    if y_train.min().item() < 0 or y_train.max().item() >= JUMLAH_KELAS:
        raise ValueError(
            f"Label harus berada pada rentang "
            f"0 sampai {JUMLAH_KELAS - 1}."
        )

    if X_train.shape[2] < 2:
        raise ValueError(
            "Panjang sequence minimal harus 2 frame "
            "untuk operasi MaxPool1d."
        )

    jumlah_fitur = X_train.shape[1]

    print("\n=== INFORMASI DATASET ===")
    print(f"Jumlah data training : {len(X_train)}")
    print(f"Jumlah data validasi : {len(X_val)}")
    print(f"Jumlah fitur         : {jumlah_fitur}")
    print(f"Bentuk input         : {tuple(X_train.shape)}")
    print(f"Jumlah kelas         : {JUMLAH_KELAS}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = buat_model(
        jumlah_fitur,
        JUMLAH_KELAS,
        nn
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_loader = DataLoader(
        TensorDataset(
            X_train,
            y_train
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n=== PROSES TRAINING CNN 1D ===")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        jumlah_data = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            output = model(X_batch)

            loss = criterion(
                output,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * len(X_batch)
            jumlah_data += len(X_batch)

        avg_loss = total_loss / jumlah_data

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f}"
        )

    # --------------------------------------------------------
    # Prediksi validation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        output_val = model(X_val)

        pred_val = (
            output_val.argmax(dim=1)
            .cpu()
            .numpy()
        )

        pred_test = None

        # Data test hanya digunakan pada evaluasi akhir
        if final and Xte is not None:

            X_test = ke_tensor(Xte, torch).to(device)

            output_test = model(X_test)

            pred_test = (
                output_test.argmax(dim=1)
                .cpu()
                .numpy()
            )

    return pred_val, pred_test


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Model CNN 1D Capstone Deep Learning NEXA."
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )

    parser.add_argument(
        "--final",
        action="store_true",
        help="Evaluasi data test pada tahap akhir."
    )

    parser.add_argument(
        "--perangkat",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Perangkat komputasi yang digunakan."
    )

    args = parser.parse_args()

    t0 = time.time()

    pred_val, pred_test = latih_dan_prediksi(
        seed=args.seed,
        final=args.final,
        perangkat=args.perangkat
    )

    durasi = time.time() - t0

    # --------------------------------------------------------
    # Menyimpan hasil prediksi
    # --------------------------------------------------------

    output_dir = AKAR / "results" / NIM

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    path_pred_val = output_dir / "pred_val.npy"

    np.save(
        path_pred_val,
        np.asarray(pred_val)
    )

    # --------------------------------------------------------
    # Evaluasi melalui protokol proyek
    # --------------------------------------------------------

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
        args.perangkat
    ]

    if pred_test is not None:

        path_pred_test = output_dir / "pred_test.npy"

        np.save(
            path_pred_test,
            np.asarray(pred_test)
        )

        perintah += [
            "--pred-test",
            str(path_pred_test)
        ]

    print("\n=== RINGKASAN EKSPERIMEN ===")
    print(f"Seed             : {args.seed}")
    print(f"Durasi training  : {durasi:.2f} detik")
    print(f"Perangkat        : {args.perangkat}")
    print(f"Jumlah prediksi validation: {len(pred_val)}")

    if pred_test is not None:
        print(f"Jumlah prediksi test: {len(pred_test)}")

    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR
    )


if __name__ == "__main__":
    main()
