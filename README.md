# Digital Theremin

Interactive webcam‑driven theremin that turns your wrist positions into pitch and volume using YOLOv8 pose detection and real‑time audio synthesis.

## Features
- Left wrist height selects notes (Bb3–Bb5) mapped to sine‑wave pitch.
- Right wrist height controls output volume.
- Live skeletal overlay with YOLOv8 pose tracking.
- Low‑latency audio stream (44.1 kHz, 512‑sample blocks).

## Requirements
- Python 3.9+ recommended.
- A webcam and working audio output.
- Dependencies listed in `requirements.txt`:
  - opencv-python
  - ultralytics (downloads `yolov8n-pose.pt` on first run)
  - mediapipe
  - sounddevice
  - numpy

## Quick Start
1) Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3) Connect your webcam, then run:
   ```bash
   python theremin.py
   ```
4) Raise/lower your **left wrist** to change pitch and your **right wrist** to change volume. Press `q` to quit.

## Internal Workings
- The YOLOv8 pose model locates keypoints each frame; the highest‑confidence person is chosen.
- Left wrist (`keypoint 9`) height is converted to a note from the predefined sequence and then to frequency.
- Right wrist (`keypoint 10`) height maps to volume (0–1).
- A streaming audio callback smoothly eases frequency and amplitude toward targets to reduce clicks.

## Tips & Troubleshooting
- If no window appears or the wrong camera is used, change the camera index in `theremin.py` (`cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)`).
- Ensure your Python environment can access the system audio device; you may need to grant microphone permission the first time.
- If audio crackles, try a larger `BLOCK_SIZE` or close other CPU‑heavy apps.
- The first run may pause briefly while `yolov8n-pose.pt` downloads.