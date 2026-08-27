# B2 Robot Vision Models — Smart Surveillance Detection Pipeline



https://github.com/user-attachments/assets/0bdb560f-835e-4942-ad84-b17086d87944


<br>

A real-time object detection pipeline built for the **Unitree B2 quadruped robot**, designed to autonomously inspect points of interest (POIs) in smart surveillance scenarios at TRX City. The system uses **YOLOX-nano** models exported to **ONNX** format to detect and classify the state of bins, electrical cabinets, and firehose cabinets from the robot's onboard cameras.

---

## Data Collection & Annotation

All training data was collected from the Unitree B2 robot during live inspection runs:

1. **ROS 2 bag recording** — Raw camera feeds were captured as ROS 2 bag files (`.db3`) by subscribing to the B2's camera topics (e.g. `/camera/cam360_*/image_raw`). The bags contain timestamped image streams from the robot's 360° and front-facing cameras.

2. **Bag-to-video conversion** — The ROS bags were converted to MP4 video files using `convert_bags_to_mp4.py`, which extracts image frames from the bag's camera topics and re-encodes them at their native frame rate.

3. **Frame extraction & annotation on Roboflow** — Key frames were extracted from the MP4 videos and uploaded to **Roboflow** for annotation. Each frame was manually labelled with bounding boxes and the following class labels:

   | Task | Classes |
   |---|---|
   | Bin fullness | `bin_empty`, `bin_full` |
   | Electrical cabinet | `cab_closed`, `cab_open` |
   | Firehose cabinet | `hose_closed`, `hose_open` |

4. **Dataset export** — Annotated datasets were exported from Roboflow in YOLO format, ready for training.

---

## Model Training

All models share the same architecture and training setup:

| Property | Detail |
|---|---|
| **Architecture** | **YOLOX-nano** — a lightweight, anchor-free single-stage object detector with a **CSPDarknet** backbone and **YOLOPAFPN** feature pyramid neck |
| **Framework** | PyTorch 2.13.0 + CUDA 13.0 |
| **Input size** | 416 × 416 (BGR, raw pixel values 0–255, no normalisation) |
| **Output** | 3,549 anchor predictions (strides 8/16/32) with objectness, class scores, and bounding box offsets |
| **Classes per model** | 2 (binary classification per task) |
| **Export format** | ONNX (opset 18) for cross-platform inference via ONNX Runtime |

YOLOX-nano was chosen for its small footprint (~4.3 MB per model) and real-time inference speed, making it ideal for deployment on the robot's companion compute (e.g. Jetson Orin) without requiring a heavy GPU.

### Trained models

| Model file | Task | Classes |
|---|---|---|
| `bin_fullness_nano_v3.onnx` | Bin fullness detection | `bin_empty`, `bin_full` |
| `electrical_cabinet_nano_v2.onnx` | Electrical cabinet state | `cab_closed`, `cab_open` |
| `firehose_cabinet_nano_v2.onnx` | Firehose cabinet state | `hose_closed`, `hose_open` |
| `bin_fullness_nano.onnx` | Original prototype (kept for reference) | `bin_empty`, `bin_full` |

---

## Inference Pipeline

The detection pipeline works as follows:

```
  Raw frame (any size)
       │
       ▼
  Letterbox resize to 416×416 (pad with grey, keep aspect ratio)
       │
       ▼
  YOLOX-nano ONNX inference (ONNX Runtime, CPU or CUDA)
       │
       ▼
  Decode raw predictions → grid decode box coordinates, apply objectness × class scores
       │
       ▼
  Confidence threshold (default 0.35) + NMS (IoU 0.45)
       │
       ▼
  Map boxes back to original image coordinates (undo letterbox)
       │
       ▼
  Per-frame detections: [{label, confidence, box}, ...]
       │
       ▼
  (Video only) Verdict system: weighted voting across frames → final answer
```

For video sources, the **verdict system** collapses all per-frame detections into a single final answer (e.g. *"HOSE_OPEN, 52% confidence, seen in 59 of 386 frames"*) using confidence-weighted majority voting, so you don't have to manually review every frame.

---

## Project Structure

```
B2_POI/
├── detect.py              ← core detection engine (CLI, single model)
├── detect_all.py          ← run all models on one source simultaneously
├── detect_gui.py          ← GUI front-end (Tkinter)
├── requirements.txt
├── README.md
└── models/
    ├── bin_fullness_nano_v3.onnx
    ├── electrical_cabinet_nano_v2.onnx
    ├── firehose_cabinet_nano_v2.onnx
    ├── bin_fullness_nano.onnx
    └── bin_fullness_nano.onnx.data
```

`detect_all.py` and `detect_gui.py` both import from `detect.py` — there is only one copy of the detection logic.

---

## Setup

Requires **Python 3.9+**.

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `onnxruntime` | Runs the ONNX models |
| `opencv-python` | Image/video I/O and drawing |
| `numpy` | Array operations for the decode step |
| `pillow` | Image display in the GUI |

---

## How to Run

### GUI (easiest)

```bash
python detect_gui.py
```

Pick a model, pick an image or video, click **Run Detection**. Done.

### CLI — single model

```bash
# Image
python detect.py --model models/firehose_cabinet_nano_v2.onnx --classes "hose_closed,hose_open" --source photo.jpg

# Video
python detect.py --model models/firehose_cabinet_nano_v2.onnx --classes "hose_closed,hose_open" --source clip.mp4 --out results/annotated.mp4

# Live camera
python detect.py --source 0 --show
```

### CLI — all models at once

```bash
python detect_all.py --source clip.mp4 --out results/multi.mp4 --json results/multi.json
```

Each model gets its own colour (red = bin, blue = cabinet, green = firehose) and its own independent verdict.

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--source` | *required* | Image, folder, video, or camera index (`0`) |
| `--model` | `models/bin_fullness_nano.onnx` | Path to `.onnx` model |
| `--classes` | `bin_empty,bin_full` | Comma-separated class names |
| `--conf` | `0.35` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--stride` | `1` | Video: run model every Nth frame (for speed) |
| `--min-frames` | `1` | Video: minimum frames for a trusted verdict |
| `--out` | — | Output folder (images) or `.mp4` path (video) |
| `--json` | — | Save detections + verdict as JSON |
| `--show` | off | Open a live preview window |
| `--gpu` | off | Use CUDA if `onnxruntime-gpu` is installed |

### Python API

```python
from detect import BinFullnessDetector, build_verdict
import cv2

det = BinFullnessDetector("models/firehose_cabinet_nano_v2.onnx",
                          class_names=("hose_closed", "hose_open"))

frame = cv2.imread("photo.jpg")
detections = det(frame)
for d in detections:
    print(d["label"], d["confidence"], d["box"])
```
