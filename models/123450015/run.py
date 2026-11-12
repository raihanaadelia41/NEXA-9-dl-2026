"""Kerangka run.py anggota. Salin folder ini menjadi models/<nim>/ lalu isi bagian MODEL.

    python models/<nim>/run.py --seed 1 [--final]
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

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "tl"                                   # cnn | tl | rnn | trf | ae


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None). Ganti isi fungsi ini dengan model Anda."""
    import torch
    import torchvision.models as models

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Konfigurasi perangkat (GPU jika ada, sebaliknya CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    
    # ---- PENYESUAIAN TENSOR UNTUK RESNET ----
    def ke_tensor_gambar(X):
        # ResNet membutuhkan input berbentuk (Batch, Channels, Height, Width).
        # Jika output data.muat() bentuknya (Batch, Height, Width, Channels), 
        # kamu perlu melakukan permute: X_tensor = X_tensor.permute(0, 3, 1, 2)
        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32))
        
        # Hapus bagian .reshape(len(X), -1) karena ResNet butuh matriks 2D/3D utuh, bukan data pipih (flatten).
        return X_tensor

    X = ke_tensor_gambar(Xtr).to(device)
    y = torch.tensor(np.asarray(ytr)).to(device)

    # ---- MODEL TRANSFER LEARNING (RESNET) ----
    # 1. Muat ResNet pre-trained (misal ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. Bekukan parameter backbone agar tidak ikut dilatih (Freeze)
    for param in model.parameters():
        param.requires_grad = False

    # 3. Ganti lapisan Fully Connected (fc) terakhir sesuai datasetmu
    num_ftrs = model.fc.in_features
    jumlah_kelas = 10 # UBAH ANGKA INI SESUAI JUMLAH KELAS PADA DATASETMU
    model.fc = torch.nn.Linear(num_ftrs, jumlah_kelas)
    
    model = model.to(device)

    # Hanya parameter di layer fc baru yang dimasukkan ke optimizer
    opt = torch.optim.SGD(model.fc.parameters(), lr=0.01)
    
    # ---- SIKLUS PELATIHAN (TRAINING LOOP) ----
    epochs = 10
    batch_size = 128
    
    model.train()
    for _ in range(epochs):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            
            opt.zero_grad()
            pred = model(X_batch)
            loss = torch.nn.functional.cross_entropy(pred, y_batch)
            loss.backward()
            opt.step()

    # ---- EVALUASI (PREDIKSI) ----
    model.eval()
    with torch.no_grad():
        # Memproses validasi dalam batch agar memori tidak penuh jika data besar
        Xva_tensor = ke_tensor_gambar(Xva).to(device)
        pred_val_list = []
        for i in range(0, len(Xva_tensor), batch_size):
            batch_pred = model(Xva_tensor[i:i + batch_size]).argmax(1).cpu().numpy()
            pred_val_list.append(batch_pred)
        pred_val = np.concatenate(pred_val_list)

        # Proses data uji
        if final and Xte is not None:
            Xte_tensor = ke_tensor_gambar(Xte).to(device)
            pred_test_list = []
            for i in range(0, len(Xte_tensor), batch_size):
                batch_pred = model(Xte_tensor[i:i + batch_size]).argmax(1).cpu().numpy()
                pred_test_list.append(batch_pred)
            pred_test = np.concatenate(pred_test_list)
        else:
            pred_test = None

    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
        
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()