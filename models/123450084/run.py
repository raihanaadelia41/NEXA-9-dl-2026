import os
import sys
import subprocess
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Tambahkan root repo ke sys.path agar src.protokol.data bisa diimpor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.protokol.data import muat


class SmallTimeSeriesTransformer(nn.Module):
    """Encoder Transformer kecil untuk urutan (pose + PIR + sensor pintu)."""
    def __init__(self, input_dim, num_classes=6, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        T = x.size(1)
        h = self.input_proj(x) + self.pos_emb[:, :T, :]
        h = self.encoder(h)
        h = h.mean(dim=1)  # Global average pooling sepanjang waktu
        return self.head(h)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    t_mulai = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    parser.add_argument("--sesi_dir", type=str, default="data/sesi_tim09")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--perangkat", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Muat data sesuai protokol
    X_train, y_train, X_val, y_val, X_test, y_test = muat(sesi_dir=args.sesi_dir)
    input_dim = X_train.shape[-1]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # 2. Bangun model Transformer kecil
    model = SmallTimeSeriesTransformer(input_dim=input_dim, num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 3. Latih model
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # 4. Prediksi validasi (dan uji hanya jika --final)
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val).to(device)).argmax(dim=1).cpu().numpy()
        test_preds = (
            model(torch.tensor(X_test).to(device)).argmax(dim=1).cpu().numpy()
            if args.final and X_test is not None
            else None
        )

    # 5. Simpan pred_val.npy dan evaluasi lewat evaluator tim (src.common.evaluate)
    durasi_detik = round(time.time() - t_mulai, 1)
    perangkat = args.perangkat or ("cuda" if torch.cuda.is_available() else "cpu laptop")

    akar = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    nim = "123450084"
    out_dir = os.path.join(akar, "results", nim)
    os.makedirs(out_dir, exist_ok=True)
    pred_val_path = os.path.join(out_dir, "pred_val.npy")
    np.save(pred_val_path, np.asarray(val_preds))

    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", nim,
        "--model", "trf",
        "--seed", str(args.seed),
        "--pred-val", pred_val_path,
        "--durasi", f"{durasi_detik:.1f}",
        "--perangkat", perangkat,
    ]
    if test_preds is not None:
        pred_test_path = os.path.join(out_dir, "pred_test.npy")
        np.save(pred_test_path, np.asarray(test_preds))
        perintah += ["--pred-test", pred_test_path]

    subprocess.run(perintah, check=True, cwd=akar)


if __name__ == "__main__":
    main()