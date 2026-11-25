
import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/processed/pose/P01/P01_block01_pose.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/quality/P01"
)

LANDMARKS = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def to_number(value):
    """Mengubah nilai menjadi angka; mengembalikan None jika tidak valid."""
    if value is None or value.strip() == "":
        return None

    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def check_quality(input_path, output_dir, threshold):
    if not input_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_results = []
    landmark_stats = {
        name: {
            "visibility_values": [],
            "low_visibility": 0,
            "missing_coordinates": 0,
            "invalid_coordinates": 0,
            "missing_visibility": 0,
            "total_frames": 0,
        }
        for name in LANDMARKS
    }

    total_frames = 0
    detected_frames = 0
    total_low_visibility = 0
    total_missing_coordinates = 0
    total_invalid_coordinates = 0

    with input_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required_columns = {"frame_index", "timestamp_ms", "pose_detected"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("Kolom wajib tidak lengkap pada file input.")

        for row in reader:
            total_frames += 1
            missing_coordinates = 0
            invalid_coordinates = 0
            low_visibility = 0
            visibility_values = []

            for landmark in LANDMARKS:
                stats = landmark_stats[landmark]
                stats["total_frames"] += 1

                coordinates = [
                    row.get(f"{landmark}_x", ""),
                    row.get(f"{landmark}_y", ""),
                    row.get(f"{landmark}_z", ""),
                ]

                for value in coordinates:
                    number = to_number(value)

                    if value is None or value.strip() == "":
                        missing_coordinates += 1
                        stats["missing_coordinates"] += 1
                    elif number is None:
                        invalid_coordinates += 1
                        stats["invalid_coordinates"] += 1

                visibility_raw = row.get(f"{landmark}_visibility", "")
                visibility = to_number(visibility_raw)

                if visibility is None:
                    stats["missing_visibility"] += 1
                else:
                    visibility_values.append(visibility)
                    stats["visibility_values"].append(visibility)

                    if visibility < threshold:
                        low_visibility += 1
                        stats["low_visibility"] += 1

            pose_detected = row.get("pose_detected", "").strip().lower()
            pose_detected = pose_detected in ("true", "1", "yes")

            if pose_detected:
                detected_frames += 1

            mean_visibility = (
                sum(visibility_values) / len(visibility_values)
                if visibility_values else None
            )

            frame_results.append({
                "frame_index": row.get("frame_index", ""),
                "timestamp_ms": row.get("timestamp_ms", ""),
                "pose_detected": pose_detected,
                "missing_coordinate_values": missing_coordinates,
                "invalid_coordinate_values": invalid_coordinates,
                "low_visibility_landmarks": low_visibility,
                "mean_visibility": (
                    round(mean_visibility, 4)
                    if mean_visibility is not None else ""
                ),
            })

            total_low_visibility += low_visibility
            total_missing_coordinates += missing_coordinates
            total_invalid_coordinates += invalid_coordinates

    # Simpan hasil pemeriksaan per frame
    frame_output = output_dir / f"{input_path.stem}_frame_qc.csv"

    frame_fields = [
        "frame_index",
        "timestamp_ms",
        "pose_detected",
        "missing_coordinate_values",
        "invalid_coordinate_values",
        "low_visibility_landmarks",
        "mean_visibility",
    ]

    with frame_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(frame_results)

    # Simpan hasil pemeriksaan per landmark
    landmark_output = output_dir / f"{input_path.stem}_landmark_qc.csv"

    landmark_fields = [
        "landmark",
        "mean_visibility",
        "low_visibility_count",
        "low_visibility_percentage",
        "missing_coordinate_values",
        "invalid_coordinate_values",
        "missing_visibility",
    ]

    with landmark_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=landmark_fields)
        writer.writeheader()

        for name, stats in landmark_stats.items():
            values = stats["visibility_values"]
            total = stats["total_frames"]

            writer.writerow({
                "landmark": name,
                "mean_visibility": (
                    round(sum(values) / len(values), 4)
                    if values else ""
                ),
                "low_visibility_count": stats["low_visibility"],
                "low_visibility_percentage": (
                    round(stats["low_visibility"] / total * 100, 2)
                    if total else 0
                ),
                "missing_coordinate_values": stats["missing_coordinates"],
                "invalid_coordinate_values": stats["invalid_coordinates"],
                "missing_visibility": stats["missing_visibility"],
            })

    # Tampilkan ringkasan
    detection_rate = (
        detected_frames / total_frames * 100 if total_frames else 0
    )

    print("\n=== HASIL QUALITY CHECK ===")
    print(f"File input                 : {input_path.name}")
    print(f"Total frame                : {total_frames}")
    print(f"Frame dengan pose          : {detected_frames}")
    print(f"Tingkat deteksi            : {detection_rate:.2f}%")
    print(f"Koordinat kosong           : {total_missing_coordinates}")
    print(f"Koordinat tidak valid      : {total_invalid_coordinates}")
    print(f"Landmark visibility < {threshold}: {total_low_visibility}")
    print(f"\nFile QC per frame          : {frame_output}")
    print(f"File QC per landmark       : {landmark_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pemeriksaan kualitas hasil ekstraksi pose."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Lokasi file CSV hasil ekstraksi pose.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder penyimpanan hasil quality check.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ambang visibility (default: 0.5).",
    )

    args = parser.parse_args()

    check_quality(
        input_path=args.input,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
