"""Transfer Learning (ResNet-18) untuk klasifikasi 6 lintasan arah gerak.

Masukan model : urutan pose + PIR + sensor pintu  →  shape (N, 10, 32)
Pemetaan ke citra: (N, 10, 32) → 3-channel 64×64 agar cocok dengan ResNet-18.
Strategi TL    : bekukan semua layer kecuali fc terakhir (fine-tune hanya head).

Jalankan:
    python models/123450015/run.py --seed 1
    python models/123450015/run.py --seed 1 --final    # sprint terakhir, sekali
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

NIM   = Path(__file__).resolve().parent.name   # otomatis: 123450015
JALUR = "tl"                                   # jalur Transfer Learning


# Konversi sekuens → citra 3-channel untuk ResNet

def _ke_citra(X: np.ndarray, ukuran: int = 64) -> "torch.Tensor":
    """(N, T, F) → (N, 3, ukuran, ukuran) — normalisasi min-max per sampel."""
    import torch
    import torch.nn.functional as F

    arr = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
    # Normalisasi per-sampel ke [0, 1]
    mn = arr.min(axis=(1, 2), keepdims=True)
    mx = arr.max(axis=(1, 2), keepdims=True)
    arr = np.where(mx > mn, (arr - mn) / (mx - mn + 1e-8), arr)

    # (N, T, F) → (N, 1, T, F) → resize → (N, 1, ukuran, ukuran)
    t = torch.tensor(arr).unsqueeze(1)                      # (N, 1, T, F)
    t = F.interpolate(t, size=(ukuran, ukuran),
                      mode="bilinear", align_corners=False)  # (N, 1, 64, 64)
    t = t.repeat(1, 3, 1, 1)                                # (N, 3, 64, 64)
    return t


# Model: ResNet-18 pretrained, hanya fc yang dilatih

def _bangun_model(jumlah_kelas: int) -> "torch.nn.Module":
    """ResNet-18 dengan fc head diganti sesuai jumlah kelas."""
    import torchvision.models as tv

    model = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT)

    # Bekukan semua layer backbone
    for param in model.parameters():
        param.requires_grad = False

    # Ganti head → trainable
    in_features = model.fc.in_features
    model.fc = __import__("torch").nn.Linear(in_features, jumlah_kelas)
    # model.fc sudah requires_grad=True secara bawaan

    return model


# Latih & prediksi

def latih_dan_prediksi(
    seed: int, final: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TL-ResNet18] perangkat={device}, seed={seed}")

    # 1. Muat data protokol tim
    X_tr, y_tr, X_va, y_va, X_te, y_te = data.muat()

    # 2. Konversi ke tensor citra
    Xtr = _ke_citra(X_tr).to(device)
    ytr = torch.tensor(np.asarray(y_tr, dtype=np.int64)).to(device)
    Xva = _ke_citra(X_va).to(device)

    jumlah_kelas = int(len(np.unique(np.asarray(y_tr))))
    print(f"  jumlah kelas={jumlah_kelas}, "
          f"n_train={len(Xtr)}, n_val={len(Xva)}")

    # 3. Bangun model ResNet-18 (TL)
    model = _bangun_model(jumlah_kelas).to(device)

    # 4. Latih hanya fc head
    opt    = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    EPOCHS = 20
    BATCH  = 32

    model.train()
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(len(Xtr), device=device)
        ep_loss = 0.0
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i: i + BATCH]
            opt.zero_grad()
            logit = model(Xtr[idx])
            loss  = F.cross_entropy(logit, ytr[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}/{EPOCHS}  loss={ep_loss / len(Xtr):.4f}")

    # 5. Prediksi
    model.eval()
    with torch.no_grad():
        pred_val_parts = []
        for i in range(0, len(Xva), BATCH):
            logit = model(Xva[i: i + BATCH])
            pred_val_parts.append(logit.argmax(1).cpu().numpy())
        pred_val = np.concatenate(pred_val_parts)

        pred_test: np.ndarray | None = None
        if final and X_te is not None:
            Xte = _ke_citra(X_te).to(device)
            pred_test_parts = []
            for i in range(0, len(Xte), BATCH):
                logit = model(Xte[i: i + BATCH])
                pred_test_parts.append(logit.argmax(1).cpu().numpy())
            pred_test = np.concatenate(pred_test_parts)

    return pred_val, pred_test



# Entry point

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed",      type=int, required=True,
                    help="seed pengacakan (mis. 1, 2, 3)")
    ap.add_argument("--final",     action="store_true",
                    help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop",
                    help="label perangkat (mis. 'RTX 3060 laptop')")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    # Simpan prediksi
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    # Panggil evaluator tim (src.common.evaluate)
    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim",        NIM,
        "--model",      JALUR,
        "--seed",       str(a.seed),
        "--pred-val",   str(out / "pred_val.npy"),
        "--durasi",     f"{durasi:.1f}",
        "--perangkat",  a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
