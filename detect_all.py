"""
Multi-model detection - run every configured model against the same source
in a single pass (one video decode, not four).

    python detect_all.py --source photo.jpg --out results/
    python detect_all.py --source clip.mp4 --out results/annotated.mp4
    python detect_all.py --source 0 --show

Each model gets its own verdict, its own colour, and its own row in the
combined summary. Detections from all models are drawn onto the same frame,
so you see everything the robot's whole vision stack would see at once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from detect import BinFullnessDetector, build_verdict, print_verdict, resolve_source, IMAGE_EXTS

# name -> (model file relative to --models-dir, class names, box colour BGR)
MODEL_REGISTRY = {
    "bin":       ("bin_fullness_nano_v3.onnx",      ("bin_empty", "bin_full"),       (0, 0, 255)),   # red
    "cabinet":   ("electrical_cabinet_nano_v2.onnx", ("cab_closed", "cab_open"),      (255, 0, 0)),   # blue
    "firehose":  ("firehose_cabinet_nano_v2.onnx",   ("hose_closed", "hose_open"),    (0, 255, 0)),   # green
}


def load_models(models_dir: Path, only: list[str] | None, conf: float, iou: float):
    names = only or list(MODEL_REGISTRY.keys())
    loaded = {}
    for name in names:
        if name not in MODEL_REGISTRY:
            sys.exit("Unknown model '%s'. Known: %s" % (name, list(MODEL_REGISTRY.keys())))
        fname, classes, color = MODEL_REGISTRY[name]
        path = models_dir / fname
        if not path.exists():
            sys.exit("Model file not found: %s" % path)
        det = BinFullnessDetector(str(path), classes, conf, iou)
        loaded[name] = {"detector": det, "color": color}
        print("Loaded %-10s -> %s  classes %s" % (name, fname, list(classes)))
    return loaded


def draw_multi(frame, results_by_model):
    """Draw every model's boxes on one frame, colour-coded, model name tag included."""
    out = frame.copy()
    thick = max(1, round(0.002 * max(out.shape[:2])))
    for name, (dets, color) in results_by_model.items():
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            text = "%s: %s %.2f" % (name, d["label"], d["confidence"])
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1, th + 4)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 2, ty), color, -1)
            cv2.putText(out, text, (x1 + 1, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_frame(frame, models):
    """Run every model on one frame. Returns {name: (dets, color)}."""
    out = {}
    for name, m in models.items():
        out[name] = (m["detector"](frame), m["color"])
    return out


def run_images(models, files, out_dir, show):
    all_results = []
    per_model_frames = {name: [] for name in models}

    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            print("  skip (unreadable): %s" % f)
            continue
        t0 = time.perf_counter()
        by_model = run_frame(frame, models)
        ms = (time.perf_counter() - t0) * 1000

        print("%s  (%.0f ms)" % (f.name, ms))
        entry = {"source": str(f), "models": {}}
        for name, (dets, _) in by_model.items():
            per_model_frames[name].append(dets)
            top = max(dets, key=lambda d: d["confidence"]) if dets else None
            tag = "%s %.2f" % (top["label"], top["confidence"]) if top else "nothing found"
            print("    %-10s -> %s  (%d box(es))" % (name, tag, len(dets)))
            entry["models"][name] = dets
        all_results.append(entry)

        if out_dir or show:
            vis = draw_multi(frame, by_model)
            if out_dir:
                cv2.imwrite(str(out_dir / ("%s_pred%s" % (f.stem, f.suffix))), vis)
            if show:
                cv2.imshow("multi-model detect", vis)
                if cv2.waitKey(0) & 0xFF in (27, ord("q")):
                    break

    print("\n" + "=" * 56)
    print("  SUMMARY over %d image(s)" % len(all_results))
    for name in models:
        s = build_verdict(per_model_frames[name])
        v = s["verdict"]
        tag = "nothing found" if v == "no_detection" else "%s (%.0f%%)" % (v, s["confidence"] * 100)
        print("    %-10s overall: %s" % (name, tag))
    print("=" * 56)
    return all_results


def run_video(models, src, out_path, show, stride, min_frames):
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

    frames = 0
    by_model = {name: ([], models[name]["color"]) for name in models}
    per_model_frames = {name: [] for name in models}
    results = []
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frames % stride == 0:
                by_model = run_frame(frame, models)
                frame_entry = {"frame": frames, "models": {}}
                for name, (dets, _) in by_model.items():
                    per_model_frames[name].append(dets)
                    frame_entry["models"][name] = dets
                results.append(frame_entry)

            vis = draw_multi(frame, by_model)
            frames += 1
            if writer:
                writer.write(vis)
            if show:
                cv2.imshow("multi-model detect", vis)
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
    print("\nProcessed %d frames in %.1fs (%.1f FPS, %d model(s) each frame)"
          % (frames, elapsed, frames / max(elapsed, 1e-6), len(models)))

    verdicts = {}
    for name in models:
        summary = build_verdict(per_model_frames[name], min_frames)
        verdicts[name] = summary
        print_verdict(summary, title="%s VERDICT" % name.upper())

    return {"verdicts": verdicts, "frames": results}


def main():
    ap = argparse.ArgumentParser(description="Run all configured models on one source at once")
    ap.add_argument("--source", required=True, help="image, folder, video file, or camera index")
    ap.add_argument("--models-dir", default="models", help="folder containing the .onnx files")
    ap.add_argument("--models", help="comma-separated subset of: %s (default: all)"
                    % ",".join(MODEL_REGISTRY.keys()))
    ap.add_argument("--out", help="output image folder or .mp4 path")
    ap.add_argument("--json", help="write all detections + verdicts to this JSON file")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--stride", type=int, default=1, help="video: infer every Nth frame")
    ap.add_argument("--min-frames", type=int, default=1,
                    help="video: how many frames must agree before a verdict is trusted")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    only = [m.strip() for m in args.models.split(",")] if args.models else None
    models = load_models(Path(args.models_dir), only, args.conf, args.iou)
    print()

    kind, src = resolve_source(args.source)
    if kind == "images":
        out_dir = Path(args.out) if args.out else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        results = run_images(models, src, out_dir, args.show)
    else:
        out_path = Path(args.out) if args.out else None
        if out_path and out_path.suffix.lower() != ".mp4":
            out_path = out_path / "annotated.mp4"
        results = run_video(models, src, out_path, args.show, max(1, args.stride),
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
