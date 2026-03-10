# Variant A: Camera Stabilization

1. reads a shaky video;
2. detects sparse feature points;
3. tracks them with Lucas-Kanade optical flow;
4. estimates global frame-to-frame camera motion with a partial affine transform;
5. smooths the camera trajectory with a moving average;
6. warps frames to the smoothed trajectory;
7. saves the results

## Files

- `src/stabilize.py` — the full pipeline
- `requirements.txt` — minimal dependencies
- `outputs/stabilized.mp4` — final stabilized video
- `outputs/comparison.mp4` — original and stabilized frames side by side
- `outputs/trajectory.png` — original and smoothed camera trajectory

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (recreating my results)

```bash
python3 src/stabilize.py --input vids/youtube_first_20s.mp4 --smooth-radius 45 --crop-ratio 0.05 --output-dir outputs_neighborhood
python3 src/stabilize.py --input vids/green.mp4 --smooth-radius 100 --crop-ratio 0.20 --max-corners 600 --output-dir outputs_green
python3 src/stabilize.py --input vids/sora.mp4 --smooth-radius 60 --crop-ratio 0.10 --max-corners 600 --output-dir outputs_sora
```

## Error Analysis

### `youtube_first_20s.mp4`

- The video contains real camera translation forward, not only hand jitter. Because of that, some residual drift and perspective change remain even after smoothing. The output is steadier, but it is not locked to a perfectly fixed viewpoint.
- The stronger center crop removes most border artifacts, but it also cuts away part of the field of view. This is a tradeoff made to avoid black edges after warping.

### `green.mp4`

- This video is a difficult case for stabilization because almost the entire frame is filled with dense green foliage. The method assumes that one global camera motion can explain most of the frame, but here there is no clear rigid background with stable structures such as buildings, horizon lines, or straight edges.
- The result is only a modest improvement. Small high-frequency shake is reduced, but the stabilized video in `outputs_green/comparison.mp4` still has visible residual wobble.
- The main failure comes from repeated leaf texture and thin branches. These features look similar across the frame, so Lucas-Kanade tracking becomes noisy and the estimated global motion is less reliable.
- Motion blur during faster camera movement makes the problem worse. In blurred frames, leaf details are less distinct, so tracking quality drops further and the output can still drift or jerk.
- The larger crop hides most border artifacts after warping, but it noticeably reduces the field of view. This is the tradeoff used to keep the final stabilized frame visually cleaner.

### `sora.mp4`

- This is the best case for the current method. The scene is almost static and dominated by rigid structures: brick walls, window frames, sidewalk edges, and a utility pole. These features give stable points for Lucas-Kanade tracking.
- The stabilization works noticeably better here
- The trajectory in `outputs_sora/trajectory.png` is also much cleaner than for the other videos. The motion curves are smooth and low-amplitude, which means the estimated camera motion is consistent and the moving-average smoothing is enough to suppress most visible jitter.
