"""Camera-only ground-truth collector for pose experiments.

Uses:
- OpenCV for camera capture and live preview
- MediaPipe Pose for keypoint extraction
- Matplotlib for post-capture visualization

Default camera index is 0 (with optional fallback to 1).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt

# MediaPipe 0.10.x can fail with protobuf>=5 because GetPrototype was removed.
# We patch SymbolDatabase to keep backward compatibility at runtime.
try:
    from google.protobuf import message_factory as _message_factory
    from google.protobuf import symbol_database as _symbol_database

    if not hasattr(_symbol_database.SymbolDatabase, "GetPrototype"):
        def _compat_get_prototype(self, descriptor):
            return _message_factory.GetMessageClass(descriptor)

        _symbol_database.SymbolDatabase.GetPrototype = _compat_get_prototype

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):
        def _compat_factory_get_prototype(self, descriptor):
            return _message_factory.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _compat_factory_get_prototype
except Exception:
    # If protobuf internals differ, we keep default behavior and let import/runtime raise normally.
    pass

import mediapipe as mp


# COCO-style 17 keypoints mapped from MediaPipe Pose landmark indices.
COCO17_MAP: List[Tuple[str, int]] = [
    ("nose", 0),
    ("left_eye", 2),
    ("right_eye", 5),
    ("left_ear", 7),
    ("right_ear", 8),
    ("left_shoulder", 11),
    ("right_shoulder", 12),
    ("left_elbow", 13),
    ("right_elbow", 14),
    ("left_wrist", 15),
    ("right_wrist", 16),
    ("left_hip", 23),
    ("right_hip", 24),
    ("left_knee", 25),
    ("right_knee", 26),
    ("left_ankle", 27),
    ("right_ankle", 28),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect camera ground-truth keypoints using MediaPipe Pose."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Primary camera index")
    parser.add_argument(
        "--fallback-index",
        type=int,
        default=1,
        help="Fallback camera index if the primary camera cannot be opened",
    )
    parser.add_argument("--width", type=int, default=1920, help="Capture width")
    parser.add_argument("--height", type=int, default=1080, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Target camera FPS")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop automatically after N seconds (0 means no time limit)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ground_truth_outputs",
        help="Directory for JSON and plot output",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display plot window after capture (still saves PNG)",
    )
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> Tuple[cv2.VideoCapture, int]:
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    used_index = args.camera_index

    if not cap.isOpened() and args.fallback_index >= 0:
        cap.release()
        cap = cv2.VideoCapture(args.fallback_index, cv2.CAP_DSHOW)
        used_index = args.fallback_index

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera_index} or fallback index {args.fallback_index}."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap, used_index


def extract_coco17(landmarks, width: int, height: int) -> List[Dict[str, float]]:
    keypoints = []
    for name, mp_idx in COCO17_MAP:
        lm = landmarks[mp_idx]
        x_px = lm.x * width
        y_px = lm.y * height
        keypoints.append(
            {
                "name": name,
                "mp_index": mp_idx,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "x_px": x_px,
                "y_px": y_px,
            }
        )
    return keypoints


def torso_length_px(coco17: List[Dict[str, float]]) -> float | None:
    by_name = {kp["name"]: kp for kp in coco17}
    rs = by_name.get("right_shoulder")
    lh = by_name.get("left_hip")
    if rs is None or lh is None:
        return None
    if rs["visibility"] < 0.1 or lh["visibility"] < 0.1:
        return None
    dx = rs["x_px"] - lh["x_px"]
    dy = rs["y_px"] - lh["y_px"]
    return math.sqrt(dx * dx + dy * dy)


def draw_red_keypoints(frame, coco17: List[Dict[str, float]], visibility_thr: float = 0.2) -> None:
    for kp in coco17:
        if kp["visibility"] < visibility_thr:
            continue
        x = int(round(kp["x_px"]))
        y = int(round(kp["y_px"]))
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)


def draw_hud(frame, used_camera: int, frame_idx: int, detected_count: int) -> None:
    # Semi-transparent background panel for text readability.
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (690, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    line1 = f"cam:{used_camera} frame:{frame_idx} det:{detected_count}/{frame_idx + 1}"
    line2 = "Press q to stop"

    def draw_outlined_text(text: str, org: Tuple[int, int], scale: float) -> None:
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    draw_outlined_text(line1, (20, 45), 0.78)
    draw_outlined_text(line2, (20, 82), 0.70)


def save_csv_outputs(frames: List[Dict], timeseries_path: Path, keypoints_path: Path) -> None:
    joints = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]

    # 1) Frame-level time series CSV with delta columns for quick trend analysis.
    header = [
        "frame_idx",
        "timestamp_sec",
        "detected",
        "torso_length_px",
        "delta_torso_length_px",
    ]
    for j in joints:
        header.extend([f"{j}_x_px", f"{j}_y_px", f"delta_{j}_y_px"])

    prev_torso = None
    prev_joint_y: Dict[str, float | None] = {j: None for j in joints}

    with open(timeseries_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for frame in frames:
            row = {
                "frame_idx": frame["frame_idx"],
                "timestamp_sec": round(frame["timestamp_sec"], 6),
                "detected": int(frame["detected"]),
                "torso_length_px": "",
                "delta_torso_length_px": "",
            }

            by_name = {
                kp["name"]: kp for kp in frame["keypoints_coco17"]
            } if frame["detected"] else {}

            torso = frame["torso_length_px"]
            if torso is not None:
                row["torso_length_px"] = round(torso, 6)
                if prev_torso is not None:
                    row["delta_torso_length_px"] = round(torso - prev_torso, 6)
                prev_torso = torso

            for j in joints:
                x_key = f"{j}_x_px"
                y_key = f"{j}_y_px"
                dy_key = f"delta_{j}_y_px"

                row[x_key] = ""
                row[y_key] = ""
                row[dy_key] = ""

                kp = by_name.get(j)
                if kp is None:
                    continue

                x = kp["x_px"]
                y = kp["y_px"]
                row[x_key] = round(x, 6)
                row[y_key] = round(y, 6)

                if prev_joint_y[j] is not None:
                    row[dy_key] = round(y - prev_joint_y[j], 6)
                prev_joint_y[j] = y

            writer.writerow(row)

    # 2) Long-format CSV for all detected COCO17 keypoints.
    kp_header = [
        "frame_idx",
        "timestamp_sec",
        "keypoint_name",
        "mp_index",
        "x",
        "y",
        "z",
        "visibility",
        "x_px",
        "y_px",
    ]
    with open(keypoints_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=kp_header)
        writer.writeheader()
        for frame in frames:
            if not frame["detected"]:
                continue
            for kp in frame["keypoints_coco17"]:
                writer.writerow(
                    {
                        "frame_idx": frame["frame_idx"],
                        "timestamp_sec": round(frame["timestamp_sec"], 6),
                        "keypoint_name": kp["name"],
                        "mp_index": kp["mp_index"],
                        "x": round(kp["x"], 6),
                        "y": round(kp["y"], 6),
                        "z": round(kp["z"], 6),
                        "visibility": round(kp["visibility"], 6),
                        "x_px": round(kp["x_px"], 6),
                        "y_px": round(kp["y_px"], 6),
                    }
                )


def build_plot(frames: List[Dict], plot_path: Path, show_plot: bool) -> None:
    detected_frames = [f for f in frames if f["detected"]]
    if not detected_frames:
        print("No detections were found. Skipping plot generation.")
        return

    idx = [f["frame_idx"] for f in detected_frames]
    torso = [f["torso_length_px"] for f in detected_frames]

    joints_to_plot = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
    joint_y: Dict[str, List[float]] = {j: [] for j in joints_to_plot}

    for f in detected_frames:
        by_name = {kp["name"]: kp for kp in f["keypoints_coco17"]}
        for j in joints_to_plot:
            if j in by_name:
                joint_y[j].append(by_name[j]["y_px"])
            else:
                joint_y[j].append(float("nan"))

    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(idx, torso, color="tab:blue", linewidth=1.8)
    ax1.set_title("Torso Length Over Time (pixels)")
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Torso Length (px)")
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    for j in joints_to_plot:
        ax2.plot(idx, joint_y[j], linewidth=1.2, label=j)
    ax2.set_title("Selected Keypoint Y-Coordinate Over Time")
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Y (px)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=140)
    print(f"Saved plot: {plot_path}")

    if show_plot:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap, used_camera = open_camera(args)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames: List[Dict] = []
    frame_idx = 0
    detected_count = 0

    start = time.time()
    print("Capture started. Press 'q' to finish recording.")
    print(f"Camera index in use: {used_camera}")

    window_name = "Camera Ground Truth (MediaPipe Pose)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed. Stopping capture.")
                break

            ts = time.time() - start
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            detected = result.pose_landmarks is not None
            frame_record: Dict = {
                "frame_idx": frame_idx,
                "timestamp_sec": ts,
                "detected": detected,
                "torso_length_px": None,
                "keypoints_coco17": [],
            }

            if detected:
                detected_count += 1
                coco17 = extract_coco17(
                    result.pose_landmarks.landmark, actual_width, actual_height
                )
                frame_record["keypoints_coco17"] = coco17
                frame_record["torso_length_px"] = torso_length_px(coco17)

                mp_draw.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=2
                    ),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(255, 150, 0), thickness=2
                    ),
                )

                # Requested display style: keypoint coordinates as red dots.
                draw_red_keypoints(frame, coco17)

            frames.append(frame_record)

            draw_hud(frame, used_camera, frame_idx, detected_count)

            cv2.imshow(window_name, frame)
            frame_idx += 1

            if args.max_seconds > 0 and ts >= args.max_seconds:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

    end = time.time()
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"ground_truth_camera_{dt_str}.json"
    timeseries_csv_path = output_dir / f"ground_truth_timeseries_{dt_str}.csv"
    keypoints_csv_path = output_dir / f"ground_truth_keypoints_{dt_str}.csv"
    plot_path = output_dir / f"ground_truth_plot_{dt_str}.png"

    payload = {
        "meta": {
            "camera_index_used": used_camera,
            "resolution": {"width": actual_width, "height": actual_height},
            "fps_target": args.fps,
            "fps_reported_by_camera": actual_fps,
            "capture_duration_sec": end - start,
            "total_frames": frame_idx,
            "detected_frames": detected_count,
            "detected_ratio": (detected_count / frame_idx) if frame_idx > 0 else 0.0,
            "note": "COCO17 keypoints extracted from MediaPipe Pose landmarks.",
        },
        "frames": frames,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved keypoints JSON: {json_path}")
    save_csv_outputs(frames, timeseries_csv_path, keypoints_csv_path)
    print(f"Saved timeseries CSV: {timeseries_csv_path}")
    print(f"Saved keypoints CSV: {keypoints_csv_path}")
    build_plot(frames, plot_path, show_plot=not args.no_plot)
    print("Done.")


if __name__ == "__main__":
    main()
