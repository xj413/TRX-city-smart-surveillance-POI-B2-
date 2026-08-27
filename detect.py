"""
Bin-fullness detection pipeline (YOLOX-nano ONNX).

Runs on images, folders of images, video files, or a live camera.

    python detect.py --source photo.jpg
    python detect.py --source ./photos --out runs/
    python detect.py --source clip.mp4 --out runs/annotated.mp4
    python detect.py --source 0 --show
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# The exported head emits raw offsets for xywh; obj/cls are already sigmoided.
STRIDES = (8, 16, 32)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
DEFAULT_CLASSES = ("bin_empty", "bin_full")


class BinFullnessDetector:
    def __init__(
        self,
        model_path,
        class_names=DEFAULT_CLASSES,
        conf_thres=0.35,
        iou_thres=0.45,
        providers=None,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=providers or ["CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = [d if isinstance(d, int) else 416 for d in inp.shape]
        self.in_h, self.in_w = shape[2], shape[3]

        n_out = self.session.get_outputs()[0].shape[-1]
        n_cls = n_out - 5 if isinstance(n_out, int) else len(class_names)
        if len(class_names) != n_cls:
            class_names = tuple("class_%d" % i for i in range(n_cls))
        self.class_names = class_names

        self._grids, self._strides = self._build_grids()
        self._colors = self._build_colors(len(class_names))

    # ---------- graph plumbing ----------

    def _build_grids(self):
        """Anchor centres and their strides, ordered to match the concatenated head."""
        grids, strides = [], []
        for stride in STRIDES:
            hsize, wsize = self.in_h // stride, self.in_w // stride
            yv, xv = np.meshgrid(np.arange(hsize), np.arange(wsize), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((hsize * wsize, 1), stride))
        return (
            np.concatenate(grids, 0).astype(np.float32),
            np.concatenate(strides, 0).astype(np.float32),
        )

    @staticmethod
    def _build_colors(n):
        hsv = np.array([[[int(180 * i / max(n, 1)), 220, 255]] for i in range(n)], np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
        return [tuple(int(c) for c in row) for row in bgr]

    # ---------- pre / post ----------

    def preprocess(self, frame):
        """Letterbox onto a 114-grey canvas, keeping aspect ratio.

        No /255 scaling and no mean/std - YOLOX consumes raw BGR pixel values.
        """
        h, w = frame.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        pad_y, pad_x = (self.in_h - nh) // 2, (self.in_w - nw) // 2

        canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(frame, (nw, nh), interpolation=interp)

        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)  # NCHW, BGR
        return np.ascontiguousarray(blob), r, (pad_x, pad_y)

    def postprocess(self, raw, ratio, pad, shape):
        pred = raw[0]
        xy = (pred[:, :2] + self._grids) * self._strides
        wh = np.exp(pred[:, 2:4]) * self._strides
        scores = pred[:, 4:5] * pred[:, 5:]

        cls_ids = scores.argmax(1)
        confs = scores[np.arange(len(scores)), cls_ids]
        keep = confs > self.conf_thres
        if not keep.any():
            return []

        xy, wh, confs, cls_ids = xy[keep], wh[keep], confs[keep], cls_ids[keep]

        # centre-xywh -> corner-xyxy, undo the letterbox, clamp to the frame
        boxes = np.empty((len(xy), 4), np.float32)
        boxes[:, :2] = xy - wh / 2
        boxes[:, 2:] = xy + wh / 2
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes /= ratio

        h, w = shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)

        dets = []
        for cid in np.unique(cls_ids):
            m = cls_ids == cid
            cb, cc = boxes[m], confs[m]
            xywh = np.stack([cb[:, 0], cb[:, 1], cb[:, 2] - cb[:, 0], cb[:, 3] - cb[:, 1]], 1)
            idx = cv2.dnn.NMSBoxes(
                xywh.tolist(), cc.astype(float).tolist(), self.conf_thres, self.iou_thres
            )
            for i in np.array(idx).reshape(-1):
                x1, y1, x2, y2 = cb[i]
                dets.append({
                    "class_id": int(cid),
                    "label": self.class_names[int(cid)],
                    "confidence": round(float(cc[i]), 4),
                    "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                })
        dets.sort(key=lambda d: -d["confidence"])
        return dets

    def __call__(self, frame):
        blob, ratio, pad = self.preprocess(frame)
        raw = self.session.run(None, {self.input_name: blob})[0]
        return self.postprocess(raw, ratio, pad, frame.shape[:2])

    # ---------- drawing ----------

    def draw(self, frame, dets):
        out = frame.copy()
        thick = max(1, round(0.002 * max(out.shape[:2])))
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            color = self._colors[d["class_id"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

            text = "%s %.2f" % (d["label"], d["confidence"])
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1, th + 4)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 2, ty), color, -1)
            cv2.putText(out, text, (x1 + 1, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out


# ---------- verdict ----------

def build_verdict(frame_dets, min_frames=1):
    """Collapse many per-frame detections into one final answer.

    Each frame votes with its single highest-confidence detection. Votes are
    weighted by confidence, so a handful of strong frames outrank a longer run of
    marginal ones. `min_frames` guards against one fluke frame deciding the call.
    """
    votes = {}
    frames_with = 0
    for dets in frame_dets:
        if not dets:
            continue
        frames_with += 1
        top = max(dets, key=lambda d: d["confidence"])
        v = votes.setdefault(top["label"], {"frames": 0, "_sum": 0.0, "best": 0.0})
        v["frames"] += 1
        v["_sum"] += top["confidence"]
        v["best"] = max(v["best"], top["confidence"])

    summary = {
        "verdict": "no_detection",
        "confidence": 0.0,
        "frames_analysed": len(frame_dets),
        "frames_with_detection": frames_with,
        "per_class": {},
    }
    if not votes:
        return summary

    for v in votes.values():
        v["avg_conf"] = round(v["_sum"] / v["frames"], 4)
        v["best"] = round(v["best"], 4)
        del v["_sum"]

    winner = max(votes, key=lambda k: votes[k]["frames"] * votes[k]["avg_conf"])
    summary["per_class"] = votes
    summary["confidence"] = votes[winner]["avg_conf"]
    summary["verdict"] = winner if votes[winner]["frames"] >= min_frames else "uncertain"
    if summary["verdict"] == "uncertain":
        summary["best_guess"] = winner
        summary["min_frames_required"] = min_frames
    return summary


def print_verdict(summary, title="FINAL VERDICT"):
    bar = "=" * 56
    print("\n" + bar)
    v = summary["verdict"]

    if v == "no_detection":
        print("  %s: NOTHING FOUND" % title)
        print("  %d frame(s) analysed, nothing above the confidence threshold."
              % summary["frames_analysed"])
        print("  Try lowering --conf (e.g. --conf 0.2).")
        print(bar)
        return

    if v == "uncertain":
        print("  %s: UNCERTAIN" % title)
        print("  Best guess '%s', but seen in only %d frame(s) - fewer than the %d required."
              % (summary["best_guess"],
                 summary["per_class"][summary["best_guess"]]["frames"],
                 summary["min_frames_required"]))
    else:
        print("  %s:  %s" % (title, v.upper()))
        print("  confidence %.0f%%   seen in %d of %d frame(s)"
              % (summary["confidence"] * 100,
                 summary["frames_with_detection"], summary["frames_analysed"]))

    if len(summary["per_class"]) > 1:
        print("  " + "-" * 40)
        print("  vote breakdown:")
        for label, c in sorted(summary["per_class"].items(), key=lambda kv: -kv[1]["frames"]):
            share = 100.0 * c["frames"] / max(summary["frames_with_detection"], 1)
            print("    %-14s %3d frame(s)  %3.0f%%  avg %.2f  best %.2f"
                  % (label, c["frames"], share, c["avg_conf"], c["best"]))
    print(bar)


# ---------- sources ----------

def resolve_source(source):
    if source.isdigit():
        return "camera", int(source)
    p = Path(source)
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit("No images found under %s" % p)
        return "images", files
    if not p.exists():
        sys.exit("Source not found: %s" % p)
    if p.suffix.lower() in VIDEO_EXTS:
        return "video", str(p)
    if p.suffix.lower() in IMAGE_EXTS:
        return "images", [p]
    sys.exit("Unsupported file type: %s" % p.suffix)


def run_images(det, files, out_dir, show):
    results = []
    tally = {}
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            print("  skip (unreadable): %s" % f)
            continue
        t0 = time.perf_counter()
        dets = det(frame)
        ms = (time.perf_counter() - t0) * 1000
        print("%s: %d detection(s) in %.0f ms" % (f.name, len(dets), ms))
        for d in dets:
            print("    %-12s %.2f  %s" % (d["label"], d["confidence"], d["box"]))

        # One image is one "frame", so its verdict is just its strongest detection.
        summary = build_verdict([dets])
        label = summary["verdict"]
        if label == "no_detection":
            print("    -> VERDICT: nothing found")
        else:
            print("    -> VERDICT: %s (%.0f%% confidence)" % (label.upper(), summary["confidence"] * 100))
        tally[label] = tally.get(label, 0) + 1
        results.append({"source": str(f), "verdict": summary, "detections": dets})

        if out_dir or show:
            vis = det.draw(frame, dets)
            if out_dir:
                cv2.imwrite(str(out_dir / ("%s_pred%s" % (f.stem, f.suffix))), vis)
            if show:
                cv2.imshow("bin fullness", vis)
                if cv2.waitKey(0) & 0xFF in (27, ord("q")):
                    break

    # A folder holds unrelated scenes, so tally them rather than voting on one answer.
    if len(tally) and sum(tally.values()) > 1:
        bar = "=" * 56
        print("\n" + bar)
        print("  SUMMARY over %d image(s)" % sum(tally.values()))
        for label, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            name = "nothing found" if label == "no_detection" else label
            print("    %-16s %3d image(s)" % (name, n))
        print(bar)
    return results


def run_video(det, src, out_path, show, stride, min_frames=1):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("Could not open video source: %s" % src)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames, dets, results, t0 = 0, [], [], time.perf_counter()
    inferred = []  # only the frames the model actually looked at, for the verdict
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # With --stride > 1 we infer on every Nth frame and reuse the last boxes in between.
            if frames % stride == 0:
                dets = det(frame)
                inferred.append(dets)
                results.append({"frame": frames, "detections": dets})
            vis = det.draw(frame, dets)
            frames += 1

            if writer:
                writer.write(vis)
            if show:
                cv2.imshow("bin fullness", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if frames % 50 == 0:
                elapsed = time.perf_counter() - t0
                print("  %d frames, %.1f FPS" % (frames, frames / max(elapsed, 1e-6)), end="\r")
    finally:
        cap.release()
        if writer:
            writer.release()

    elapsed = time.perf_counter() - t0
    print("\nProcessed %d frames in %.1fs (%.1f FPS)" % (frames, elapsed, frames / max(elapsed, 1e-6)))

    summary = build_verdict(inferred, min_frames)
    print_verdict(summary)
    return {"verdict": summary, "frames": results}


def main():
    ap = argparse.ArgumentParser(description="Bin-fullness detection (YOLOX-nano ONNX)")
    ap.add_argument("--source", required=True,
                    help="image, folder, video file, or camera index (e.g. 0)")
    ap.add_argument("--model", default="models/bin_fullness_nano.onnx")
    ap.add_argument("--out", help="output image folder or .mp4 path")
    ap.add_argument("--json", help="write detections to this JSON file")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--classes", help="comma-separated class names, in class-id order")
    ap.add_argument("--stride", type=int, default=1, help="video: infer every Nth frame")
    ap.add_argument("--min-frames", type=int, default=1,
                    help="video: how many frames must agree before the verdict is trusted")
    ap.add_argument("--show", action="store_true", help="display a preview window")
    ap.add_argument("--gpu", action="store_true", help="try CUDA, fall back to CPU")
    args = ap.parse_args()

    if not Path(args.model).exists():
        sys.exit("Model not found: %s" % args.model)

    providers = ["CPUExecutionProvider"]
    if args.gpu:
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]

    names = tuple(c.strip() for c in args.classes.split(",")) if args.classes else DEFAULT_CLASSES
    det = BinFullnessDetector(args.model, names, args.conf, args.iou, providers)
    print("Model: %s  input %dx%d  classes %s" %
          (args.model, det.in_w, det.in_h, list(det.class_names)))
    print("Provider: %s" % det.session.get_providers()[0])

    kind, src = resolve_source(args.source)
    if kind == "images":
        out_dir = Path(args.out) if args.out else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        results = run_images(det, src, out_dir, args.show)
    else:
        out_path = Path(args.out) if args.out else None
        if out_path and out_path.suffix.lower() != ".mp4":
            out_path = out_path / "annotated.mp4"
        results = run_video(det, src, out_path, args.show, max(1, args.stride),
                            max(1, args.min_frames))

    if args.show:
        cv2.destroyAllWindows()
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print("Wrote %s" % args.json)
    if args.out:
        print("Output: %s" % args.out)


if __name__ == "__main__":
    main()
