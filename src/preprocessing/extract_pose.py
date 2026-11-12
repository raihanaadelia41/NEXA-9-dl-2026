
"""
Ekstraksi pose dari satu video menggunakan MediaPipe Pose.
Output: CSV berisi 33 landmark untuk setiap frame.
"""

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp


# Konfigurasi project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "raw" / "videos" /
    "P01" / "P01_block01.mp4"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "pose" /
    "P01" / "P01_block01_pose.csv"
)

MAX_WIDTH = 640
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


def extract_pose(video_path, output_path):
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video tidak ditemukan: {video_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Video tidak dapat dibuka.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_pose = mp.solutions.pose

    # Nama kolom landmark
    landmarks = list(mp_pose.PoseLandmark)

    fieldnames = [
        "frame_index",
        "timestamp_ms",
        "pose_detected"
    ]

    for landmark in landmarks:
        name = landmark.name.lower()
        for coordinate in ["x", "y", "z", "visibility"]:
            fieldnames.append(f"{name}_{coordinate}")

    frame_index = 0
    detected_frames = 0

    print(f"Video       : {video_path.name}")
    print(f"Resolusi    : {original_width}x{original_height}")
    print(f"FPS         : {fps:.2f}")
    print(f"Total frame : {total_frames}")
    print("Memulai ekstraksi...")

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        ) as pose:

            with output_path.open(
                "w", newline="", encoding="utf-8"
            ) as csv_file:

                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=fieldnames
                )
                writer.writeheader()

                while True:
                    success, frame = cap.read()

                    if not success:
                        break

                    # Resize untuk mengurangi beban CPU
                    height, width = frame.shape[:2]

                    if width > MAX_WIDTH:
                        new_height = int(
                            height * MAX_WIDTH / width
                        )
                        frame = cv2.resize(
                            frame,
                            (MAX_WIDTH, new_height)
                        )

                    # OpenCV BGR -> RGB
                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )
                    rgb_frame.flags.writeable = False

                    results = pose.process(rgb_frame)

                    row = {
                        "frame_index": frame_index,
                        "timestamp_ms": round(
                            frame_index / fps * 1000, 2
                        ),
                        "pose_detected": 0
                    }

                    # Isi default jika pose tidak ditemukan
                    for landmark in landmarks:
                        name = landmark.name.lower()
                        for coordinate in [
                            "x", "y", "z", "visibility"
                        ]:
                            row[f"{name}_{coordinate}"] = ""

                    if results.pose_landmarks:
                        detected_frames += 1
                        row["pose_detected"] = 1

                        for landmark, point in zip(
                            landmarks,
                            results.pose_landmarks.landmark
                        ):
                            name = landmark.name.lower()

                            row[f"{name}_x"] = point.x
                            row[f"{name}_y"] = point.y
                            row[f"{name}_z"] = point.z
                            row[f"{name}_visibility"] = (
                                point.visibility
                            )

                    writer.writerow(row)
                    frame_index += 1

                    if frame_index % 100 == 0:
                        print(
                            f"Progress: {frame_index}/"
                            f"{total_frames} frame"
                        )

    finally:
        cap.release()

    detection_rate = (
        detected_frames / frame_index * 100
        if frame_index > 0 else 0
    )

    print("\n=== EKSTRAKSI SELESAI ===")
    print(f"Total frame diproses : {frame_index}")
    print(f"Pose terdeteksi      : {detected_frames}")
    print(f"Pose tidak terdeteksi: {frame_index - detected_frames}")
    print(f"Tingkat deteksi      : {detection_rate:.2f}%")
    print(f"File output          : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ekstraksi pose dari video menggunakan MediaPipe."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Lokasi file video"
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Lokasi file CSV hasil ekstraksi"
    )

    args = parser.parse_args()

    extract_pose(args.input, args.output)