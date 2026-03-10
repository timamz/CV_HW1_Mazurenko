from pathlib import Path
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def moving_average(values, radius):
    if radius <= 0:
        return values.copy()
    window = 2 * radius + 1
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_trajectory(trajectory, radius):
    smoothed = np.zeros_like(trajectory)
    for i in range(trajectory.shape[1]):
        smoothed[:, i] = moving_average(trajectory[:, i], radius)
    return smoothed
            

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def crop_and_resize(frame, crop_ratio):
    if crop_ratio <= 0:
        return frame
    h, w = frame.shape[:2]
    dx = int(w * crop_ratio / 2)
    dy = int(h * crop_ratio / 2)
    if dx <= 0 or dy <= 0 or 2 * dx >= w or 2 * dy >= h:
        return frame
    cropped = frame[dy:h - dy, dx:w - dx]
    return cv2.resize(cropped, (w, h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--smooth-radius", type=int, default=15)
    parser.add_argument("--max-corners", type=int, default=200)
    parser.add_argument("--crop-ratio", type=float, default=0.12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, prev = cap.read()
    if not ok:
        raise SystemExit("Cannot read first frame")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = []

    for _ in tqdm(range(frame_count - 1), desc="Estimating motion"):
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=args.max_corners,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=3,
        )

        ok, curr = cap.read()
        if not ok:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        if prev_pts is None or len(prev_pts) < 3:
            transforms.append([0.0, 0.0, 0.0])
            prev_gray = curr_gray
            continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is None or status is None:
            transforms.append([0.0, 0.0, 0.0])
            prev_gray = curr_gray
            continue

        idx = status.flatten() == 1
        prev_matched = prev_pts[idx]
        curr_matched = curr_pts[idx]

        if len(prev_matched) < 3:
            transforms.append([0.0, 0.0, 0.0])
            prev_gray = curr_gray
            continue

        matrix, _ = cv2.estimateAffinePartial2D(prev_matched, curr_matched)
        if matrix is None:
            transforms.append([0.0, 0.0, 0.0])
            prev_gray = curr_gray
            continue

        dx = matrix[0, 2]
        dy = matrix[1, 2]
        da = np.arctan2(matrix[1, 0], matrix[0, 0])
        transforms.append([dx, dy, da])
        prev_gray = curr_gray

    cap.release()

    transforms = np.array(transforms, dtype=np.float32)
    if len(transforms) == 0:
        raise SystemExit("Not enough frames to stabilize")

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory, args.smooth_radius)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap = cv2.VideoCapture(args.input)
    ok, first = cap.read()
    if not ok:
        raise SystemExit("Cannot reread input video")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    stabilized_writer = cv2.VideoWriter(str(output_dir / "stabilized.mp4"), fourcc, fps, (width, height))
    comparison_writer = cv2.VideoWriter(str(output_dir / "comparison.mp4"), fourcc, fps, (width * 2, height))

    stabilized_first = first.copy()
    stabilized_first = crop_and_resize(stabilized_first, args.crop_ratio)
    stabilized_writer.write(stabilized_first)
    comparison_writer.write(np.hstack([first, stabilized_first]))

    for i in tqdm(range(len(transforms_smooth)), desc="Writing video"):
        ok, frame = cap.read()
        if not ok:
            break

        dx, dy, da = transforms_smooth[i]
        cos = np.cos(da)
        sin = np.sin(da)
        matrix = np.array([[cos, -sin, dx], [sin, cos, dy]], dtype=np.float32)
        stabilized = cv2.warpAffine(frame, matrix, (width, height))
        stabilized = crop_and_resize(stabilized, args.crop_ratio)

        stabilized_writer.write(stabilized)
        comparison = np.hstack([frame, stabilized])
        comparison_writer.write(comparison)

    cap.release()
    stabilized_writer.release()
    comparison_writer.release()

    plt.figure(figsize=(10, 6))
    labels = ["x", "y", "angle"]
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(trajectory[:, i], label="original")
        plt.plot(smoothed_trajectory[:, i], label="smoothed")
        plt.ylabel(labels[i])
        plt.legend()
    plt.xlabel("frame")
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
