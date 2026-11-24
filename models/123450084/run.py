import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sesi_dir", type=str, default="data/sesi_tim09")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    # 4. Evaluasi F1-makro pada validasi & uji
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val).to(device)).argmax(dim=1).cpu().numpy()
        test_preds = model(torch.tensor(X_test).to(device)).argmax(dim=1).cpu().numpy()

    f1_val = float(f1_score(y_val, val_preds, average="macro"))
    f1_test = float(f1_score(y_test, test_preds, average="macro"))

    # 5. Baca data_hash dari MANIFEST.json jika ada
    data_hash = "ISI_DENGAN_SHA256_LABEL_CSV"
    manifest_path = "data/MANIFEST.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as mf:
            data_hash = json.load(mf).get("data_hash", data_hash)

    # 6. Simpan results/123450084/metrics.json
    nim = "123450084"
    out_dir = os.path.join("results", nim)
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        "nim": nim,
        "model": "Transformer",
        "seed": args.seed,
        "f1_macro_val": round(f1_val, 4),
        "f1_macro_test": round(f1_test, 4),
        "data_hash": data_hash,
    }
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"[SELESAI] F1-makro Val: {f1_val:.4f} | Test: {f1_test:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
