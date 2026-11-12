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

NIM = Path(__file__).resolve().parent.name     # otomatis terbaca: 123450015
JALUR = "tl"                                   # jalur Transfer Learning


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Transfer Learning (MobileNetV3-Small) untuk data sekuensial pose."""
    import torch
    import torch.nn.functional as F
    import torchvision.models as models

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Muat data dari protokol tim
    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    # 2. Ubah matriks sekuensial pose (Batch, Frame, Fitur) menjadi citra 2D 3-channel (64x64)
    def ke_tensor_pose_2d(X):
        arr = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
        t = torch.tensor(arr)
        if t.ndim == 2:
            t = t.unsqueeze(1)
        # Bentuk (Batch, 1, Frame, Fitur) -> duplikasi ke 3 channel RGB -> resize ke 64x64
        t = t.unsqueeze(1).repeat(1, 3, 1, 1)
        t = F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)
        return t

    X_train = ke_tensor_pose_2d(Xtr).to(device)
    y_train = torch.tensor(np.asarray(ytr), dtype=torch.long).to(device)
    jumlah_kelas = int(y_train.max().item()) + 1

    # 3. Bangun Model Transfer Learning (MobileNetV3-Small)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Bekukan bobot ekstraktor fitur bawaan (backbone)
    for param in model.features.parameters():
        param.requires_grad = False

    # Ganti layer klasifikasi terakhir agar sesuai jumlah kelas gerakan
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, jumlah_kelas)
    model = model.to(device)

    # 4. Latih bagian classifier
    opt = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    epochs = 15
    batch_size = 64

    model.train()
    for _ in range(epochs):
        indeks = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = indeks[i:i + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()

    # 5. Prediksi data validasi (dan data uji jika --final aktif)
    model.eval()
    with torch.no_grad():
        X_val = ke_tensor_pose_2d(Xva).to(device)
        pred_val_list = [
            model(X_val[i:i + batch_size]).argmax(1).cpu().numpy()
            for i in range(0, len(X_val), batch_size)
        ]
        pred_val = np.concatenate(pred_val_list)

        if final and Xte is not None:
            X_test = ke_tensor_pose_2d(Xte).to(device)
            pred_test_list = [
                model(X_test[i:i + batch_size]).argmax(1).cpu().numpy()
                for i in range(0, len(X_test), batch_size)
            ]
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

    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()