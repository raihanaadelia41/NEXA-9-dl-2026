
"""Model CNN 1D untuk Capstone Deep Learning NEXA.

Jalankan:
    python models/123450041/run.py --seed 1

Opsi --final hanya digunakan pada evaluasi akhir sesuai protokol.
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

from src.protokol import data  

NIM = Path(__file__).resolve().parent.name
JALUR = "cnn"


def latih_dan_prediksi(
    seed: int,
    final: bool
) -> tuple[np.ndarray, np.ndarray | None]:

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    def ke_tensor(X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"Input CNN 1D harus berbentuk "
                f"(sampel, waktu, fitur). Diperoleh: {X.shape}"
            )

        return torch.tensor(X).permute(0, 2, 1).contiguous()

    X_train = ke_tensor(Xtr)

    y_train = torch.tensor(
        np.asarray(ytr),
        dtype=torch.long
    )

    jumlah_fitur = X_train.shape[1]
    jumlah_kelas = 6

    model = nn.Sequential(
        nn.Conv1d(
            in_channels=jumlah_fitur,
            out_channels=32,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),

        nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),

        nn.Flatten(),
        nn.Linear(64, jumlah_kelas)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=16,
        shuffle=True
    )

    epochs = 20

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            output = model(X_batch)

            loss = criterion(output, y_batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(X_train)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {avg_loss:.4f}"
        )

    model.eval()

    with torch.no_grad():

        X_val = ke_tensor(Xva)

        output_val = model(X_val)

        pred_val = (
            output_val.argmax(dim=1).cpu().numpy()
        )

        pred_test = None

        if final and Xte is not None:

            X_test = ke_tensor(Xte)

            output_test = model(X_test)

            pred_test = (
                output_test.argmax(dim=1).cpu().numpy()
            )

    return pred_val, pred_test


def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--seed",
        type=int,
        required=True
    )

    ap.add_argument(
        "--final",
        action="store_true",
        help="Sprint terakhir: evaluasi data uji sekali"
    )

    ap.add_argument(
        "--perangkat",
        default="cpu laptop"
    )

    a = ap.parse_args()

    t0 = time.time()

    pred_val, pred_test = latih_dan_prediksi(
        a.seed,
        a.final
    )

    durasi = time.time() - t0

    out = AKAR / "results" / NIM

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        out / "pred_val.npy",
        np.asarray(pred_val)
    )

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat
    ]

    if pred_test is not None:

        np.save(
            out / "pred_test.npy",
            np.asarray(pred_test)
        )

        perintah += [
            "--pred-test",
            str(out / "pred_test.npy")
        ]

    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR
    )


if __name__ == "__main__":
    main()
