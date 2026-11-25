import os
import tarfile
import argparse
import pandas as pd
import numpy as np

def ekstrak_tar(tar_path, extract_dir):
    """Mengekstrak berkas .tar sesi meja ke direktori tujuan."""
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_dir)
    print(f"[OK] Arsip {tar_path} berhasil diekstrak ke {extract_dir}")

def ekstrak_pose_mediapipe(sesi_dir, out_csv):
    import urllib.request
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    kamera_idx_path = os.path.join(sesi_dir, "kamera_index.csv")
    if not os.path.exists(kamera_idx_path):
        raise FileNotFoundError(f"kamera_index.csv tidak ditemukan di {sesi_dir}")

    # Unduh model pose_landmarker_lite.task jika belum ada
    model_path = os.path.join(sesi_dir, "pose_landmarker_lite.task")
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        print("[INFO] Mengunduh model MediaPipe Pose Landmarker...")
        urllib.request.urlretrieve(url, model_path)

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )

    df_cam = pd.read_csv(kamera_idx_path)
    KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
    rows = []

    with vision.PoseLandmarker.create_from_options(options) as detector:
        for _, row in df_cam.iterrows():
            img_rel = row.get("file", row.get("path", ""))
            img_path = os.path.join(sesi_dir, str(img_rel)) if not os.path.isabs(str(img_rel)) else str(img_rel)
            if not os.path.exists(img_path):
                img_path = os.path.join(sesi_dir, "kamera", os.path.basename(str(img_rel)))

            feat = {"t_host": row["t_host"]}
            img = cv2.imread(img_path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_img)
                if res.pose_landmarks and len(res.pose_landmarks) > 0:
                    lm = res.pose_landmarks[0]
                    for idx in KEYPOINTS:
                        feat[f"kp{idx}_x"] = lm[idx].x
                        feat[f"kp{idx}_y"] = lm[idx].y
                        feat[f"kp{idx}_v"] = lm[idx].visibility
                    feat["hip_cx"] = (lm[23].x + lm[24].x) / 2.0
                    feat["hip_cy"] = (lm[23].y + lm[24].y) / 2.0
                else:
                    for idx in KEYPOINTS:
                        feat[f"kp{idx}_x"] = 0.0
                        feat[f"kp{idx}_y"] = 0.0
                        feat[f"kp{idx}_v"] = 0.0
                    feat["hip_cx"] = 0.0
                    feat["hip_cy"] = 0.0
            rows.append(feat)

    df_pose = pd.DataFrame(rows)
    df_pose.to_csv(out_csv, index=False)
    print(f"[OK] Ekstraksi pose selesai ({len(df_pose)} frame) -> {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", type=str, default="data/tim09_web-20261117-171532-679f.tar")
    parser.add_argument("--sesi_dir", type=str, default="data/sesi_tim09")
    parser.add_argument("--out", type=str, default="data/sesi_tim09/pose.csv")
    args = parser.parse_args()

    if os.path.exists(args.tar) and not os.path.exists(os.path.join(args.sesi_dir, "label.csv")):
        ekstrak_tar(args.tar, args.sesi_dir)
    ekstrak_pose_mediapipe(args.sesi_dir, args.out)
