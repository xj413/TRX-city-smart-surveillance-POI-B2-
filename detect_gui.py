"""
GUI front end for detect.py - load a model, an image or video, and run.

    python detect_gui.py

No command line needed. Pick a model file, pick a source, click Run.
Reuses BinFullnessDetector and build_verdict from detect.py directly, so the
detection logic is identical to the CLI - this is just a window around it.
"""

from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from detect import BinFullnessDetector, build_verdict, IMAGE_EXTS, VIDEO_EXTS

DEFAULT_MODELS_DIR = Path("models")
PRESET_CLASSES = {
    "bin_fullness_nano_v3.onnx": "bin_empty,bin_full",
    "bin_fullness_nano.onnx": "bin_empty,bin_full",
    "electrical_cabinet_nano_v2.onnx": "cab_closed,cab_open",
    "firehose_cabinet_nano_v2.onnx": "hose_closed,hose_open",
}


class DetectGUI:
    def __init__(self, root):
        self.root = root
        root.title("Bin Fullness Detector")
        root.geometry("980x720")
        root.minsize(820, 600)

        self.model_path = tk.StringVar()
        self.source_path = tk.StringVar()
        self.classes = tk.StringVar(value="bin_empty,bin_full")
        self.conf = tk.DoubleVar(value=0.35)
        self.iou = tk.DoubleVar(value=0.45)
        self.stride = tk.IntVar(value=1)
        self.min_frames = tk.IntVar(value=1)
        self.status = tk.StringVar(value="Load a model and a source, then click Run.")

        self.detector = None
        self.busy = False
        self._preview_photo = None  # keep a reference or Tk garbage-collects it

        self._build_layout()

    # ---------------- layout ----------------

    def _build_layout(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        # Model row
        ttk.Label(top, text="Model (.onnx):", width=16).grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.model_path).grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Browse...", command=self._pick_model).grid(row=0, column=2, padx=4)

        # Classes row
        ttk.Label(top, text="Classes:", width=16).grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.classes).grid(row=1, column=1, sticky="ew")
        ttk.Label(top, text="(class_id 0 , class_id 1 - comma separated)").grid(row=1, column=2, sticky="w")

        # Source row
        ttk.Label(top, text="Image / Video:", width=16).grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.source_path).grid(row=2, column=1, sticky="ew")
        src_btns = ttk.Frame(top)
        src_btns.grid(row=2, column=2, sticky="w")
        ttk.Button(src_btns, text="Image...", command=self._pick_image).pack(side="left", padx=2)
        ttk.Button(src_btns, text="Video...", command=self._pick_video).pack(side="left", padx=2)

        top.columnconfigure(1, weight=1)

        # Parameters row
        params = ttk.Frame(self.root)
        params.pack(fill="x", **pad)

        ttk.Label(params, text="Confidence").grid(row=0, column=0)
        ttk.Scale(params, from_=0.05, to=0.95, variable=self.conf, orient="horizontal",
                  length=140).grid(row=0, column=1, padx=6)
        self.conf_lbl = ttk.Label(params, text="0.35", width=5)
        self.conf_lbl.grid(row=0, column=2)

        ttk.Label(params, text="IoU").grid(row=0, column=3, padx=(16, 0))
        ttk.Scale(params, from_=0.1, to=0.9, variable=self.iou, orient="horizontal",
                  length=140).grid(row=0, column=4, padx=6)
        self.iou_lbl = ttk.Label(params, text="0.45", width=5)
        self.iou_lbl.grid(row=0, column=5)

        ttk.Label(params, text="Video stride").grid(row=0, column=6, padx=(16, 0))
        ttk.Spinbox(params, from_=1, to=30, textvariable=self.stride, width=5).grid(row=0, column=7)

        ttk.Label(params, text="Min frames").grid(row=0, column=8, padx=(16, 0))
        ttk.Spinbox(params, from_=1, to=200, textvariable=self.min_frames, width=5).grid(row=0, column=9)

        self.conf.trace_add("write", lambda *_: self.conf_lbl.config(text="%.2f" % self.conf.get()))
        self.iou.trace_add("write", lambda *_: self.iou_lbl.config(text="%.2f" % self.iou.get()))

        # Run button + progress
        run_row = ttk.Frame(self.root)
        run_row.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run_row, text="Run Detection", command=self._on_run)
        self.run_btn.pack(side="left")
        self.progress = ttk.Progressbar(run_row, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=10)
        ttk.Label(run_row, textvariable=self.status).pack(side="left", padx=10)

        # Body: preview (left) + verdict/log (right)
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, **pad)

        self.preview = tk.Label(body, background="#222", text="Preview appears here",
                                 foreground="#888")
        self.preview.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = ttk.Frame(body, width=340)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        ttk.Label(right, text="Verdict", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.verdict_box = tk.Text(right, height=8, wrap="word", state="disabled")
        self.verdict_box.pack(fill="x", pady=(2, 8))

        ttk.Label(right, text="Log", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.log_box = tk.Text(right, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True)

        # Bottom: save buttons
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", **pad)
        save_run_btn = ttk.Button(bottom, text="Save Test Run...", command=self._save_run)
        save_run_btn.pack(side="left")
        ttk.Button(bottom, text="Save Annotated Output...", command=self._save_output).pack(side="left", padx=8)
        ttk.Button(bottom, text="Save Detections as JSON...", command=self._save_json).pack(side="left")

        self._last_result = None  # holds (kind, annotated_frame_or_None, verdict, all_detections)
        self._run_meta = None     # model / source / thresholds used for the last run

    # ---------------- pickers ----------------

    def _pick_model(self):
        start = str(DEFAULT_MODELS_DIR) if DEFAULT_MODELS_DIR.exists() else "."
        path = filedialog.askopenfilename(title="Select ONNX model", initialdir=start,
                                          filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")])
        if not path:
            return
        self.model_path.set(path)
        preset = PRESET_CLASSES.get(Path(path).name)
        if preset:
            self.classes.set(preset)
        self._log("Model selected: %s" % path)

    def _pick_image(self):
        exts = " ".join("*" + e for e in IMAGE_EXTS)
        path = filedialog.askopenfilename(title="Select an image",
                                          filetypes=[("Images", exts), ("All files", "*.*")])
        if path:
            self.source_path.set(path)
            self._log("Source selected: %s" % path)

    def _pick_video(self):
        exts = " ".join("*" + e for e in VIDEO_EXTS)
        path = filedialog.askopenfilename(title="Select a video",
                                          filetypes=[("Videos", exts), ("All files", "*.*")])
        if path:
            self.source_path.set(path)
            self._log("Source selected: %s" % path)

    # ---------------- logging helpers ----------------

    def _log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _set_verdict_text(self, text):
        self.verdict_box.config(state="normal")
        self.verdict_box.delete("1.0", "end")
        self.verdict_box.insert("1.0", text)
        self.verdict_box.config(state="disabled")

    def _show_preview(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        max_w, max_h = 640, 560
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview.config(image=self._preview_photo, text="")

    # ---------------- run ----------------

    def _on_run(self):
        if self.busy:
            return
        model_p = self.model_path.get().strip()
        src_p = self.source_path.get().strip()
        if not model_p or not Path(model_p).exists():
            messagebox.showerror("Missing model", "Pick a valid .onnx model file first.")
            return
        if not src_p or not Path(src_p).exists():
            messagebox.showerror("Missing source", "Pick a valid image or video file first.")
            return

        names = tuple(c.strip() for c in self.classes.get().split(",") if c.strip())
        if len(names) != 2:
            messagebox.showerror("Bad class names", "Enter exactly two class names, comma-separated.")
            return

        # Read every Tk variable here, on the main thread - Tk variables are not
        # thread-safe, and the worker below runs on a background thread.
        conf = self.conf.get()
        iou = self.iou.get()
        stride = max(1, self.stride.get())
        min_frames = max(1, self.min_frames.get())

        self._run_meta = {
            "model": model_p,
            "classes": list(names),
            "source": src_p,
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "video_stride": stride,
            "min_frames": min_frames,
            "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.busy = True
        self.run_btn.config(state="disabled")
        self.progress.start(12)
        self.status.set("Loading model...")
        self._set_verdict_text("")
        threading.Thread(target=self._run_worker,
                         args=(model_p, src_p, names, conf, iou, stride, min_frames),
                         daemon=True).start()

    def _run_worker(self, model_p, src_p, names, conf, iou, stride, min_frames):
        try:
            det = BinFullnessDetector(model_p, names, conf, iou)
            ext = Path(src_p).suffix.lower()
            if ext in IMAGE_EXTS:
                self._run_image(det, src_p)
            elif ext in VIDEO_EXTS:
                self._run_video(det, src_p, stride, min_frames)
            else:
                self._fail("Unsupported file type: %s" % ext)
                return
        except Exception as e:
            self._fail("Error: %s" % e)
            return
        self.root.after(0, self._run_done)

    def _run_image(self, det, src_p):
        frame = cv2.imread(src_p)
        if frame is None:
            self._fail("Could not read image: %s" % src_p)
            return
        self.root.after(0, lambda: self.status.set("Running..."))
        t0 = time.perf_counter()
        dets = det(frame)
        ms = (time.perf_counter() - t0) * 1000
        vis = det.draw(frame, dets)
        summary = build_verdict([dets])

        self._last_result = ("image", vis, summary, [{"source": src_p, "detections": dets}])
        self.root.after(0, lambda: self._show_preview(vis))
        self.root.after(0, lambda: self._log("%s analysed in %.0f ms - %d detection(s)"
                                             % (Path(src_p).name, ms, len(dets))))
        self.root.after(0, lambda: self._set_verdict_text(self._format_verdict(summary)))

    def _run_video(self, det, src_p, stride, min_frames):
        cap = cv2.VideoCapture(src_p)
        if not cap.isOpened():
            self._fail("Could not open video: %s" % src_p)
            return

        frame_dets, results, last_vis = [], [], None
        i, t0 = 0, time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    dets = det(frame)
                    frame_dets.append(dets)
                    results.append({"frame": i, "detections": dets})
                    last_vis = det.draw(frame, dets)
                    if i % 5 == 0:
                        self.root.after(0, lambda f=i, vis=last_vis: (
                            self._show_preview(vis),
                            self.status.set("Processing frame %d..." % f),
                        ))
                i += 1
        finally:
            cap.release()

        elapsed = time.perf_counter() - t0
        summary = build_verdict(frame_dets, min_frames)
        self._last_result = ("video", last_vis, summary, results)
        self.root.after(0, lambda: self._log(
            "%s: %d frames in %.1fs (%.1f FPS)" % (Path(src_p).name, i, elapsed, i / max(elapsed, 1e-6))))
        self.root.after(0, lambda: self._set_verdict_text(self._format_verdict(summary)))
        if last_vis is not None:
            self.root.after(0, lambda: self._show_preview(last_vis))

    def _format_verdict(self, summary):
        v = summary["verdict"]
        if v == "no_detection":
            return "NOTHING FOUND\n\n%d frame(s) analysed, nothing above the confidence threshold.\nTry lowering Confidence." % summary["frames_analysed"]
        lines = []
        if v == "uncertain":
            lines.append("UNCERTAIN")
            lines.append("Best guess: %s" % summary.get("best_guess"))
            bg = summary.get("best_guess")
            lines.append("Seen in %d frame(s), fewer than the %d required."
                         % (summary["per_class"][bg]["frames"], summary["min_frames_required"]))
        else:
            lines.append(v.upper())
            lines.append("Confidence: %.0f%%" % (summary["confidence"] * 100))
            lines.append("Seen in %d of %d frame(s)"
                         % (summary["frames_with_detection"], summary["frames_analysed"]))
        if len(summary["per_class"]) > 1:
            lines.append("")
            lines.append("Vote breakdown:")
            for label, c in sorted(summary["per_class"].items(), key=lambda kv: -kv[1]["frames"]):
                lines.append("  %-14s %3d frame(s)  avg %.2f  best %.2f"
                             % (label, c["frames"], c["avg_conf"], c["best"]))
        return "\n".join(lines)

    def _fail(self, msg):
        self.root.after(0, lambda: (self._log(msg), messagebox.showerror("Error", msg)))
        self.root.after(0, self._run_done)

    def _run_done(self):
        self.busy = False
        self.run_btn.config(state="normal")
        self.progress.stop()
        self.status.set("Done.")

    # ---------------- save ----------------

    def _save_output(self):
        if not self._last_result:
            messagebox.showinfo("Nothing to save", "Run detection first.")
            return
        kind, vis, _, _ = self._last_result
        if vis is None:
            messagebox.showinfo("Nothing to save", "No annotated frame available.")
            return
        default_ext = ".jpg" if kind == "image" else ".jpg"
        path = filedialog.asksaveasfilename(defaultextension=default_ext,
                                            filetypes=[("JPEG image", "*.jpg")])
        if path:
            cv2.imwrite(path, vis)
            self._log("Saved annotated frame -> %s" % path)

    def _save_json(self):
        if not self._last_result:
            messagebox.showinfo("Nothing to save", "Run detection first.")
            return
        kind, _, summary, detections = self._last_result
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        payload = {"verdict": summary, "detections": detections}
        Path(path).write_text(json.dumps(payload, indent=2))
        self._log("Saved detections -> %s" % path)

    def _save_run(self):
        """Save everything from the last run - annotated frame, full JSON, and a
        readable text report - into one timestamped folder, so a test run can be
        archived and compared against others later."""
        if not self._last_result or not self._run_meta:
            messagebox.showinfo("Nothing to save", "Run detection first.")
            return

        kind, vis, summary, detections = self._last_result
        meta = self._run_meta

        default_name = "run_%s_%s" % (
            time.strftime("%Y%m%d_%H%M%S"),
            Path(meta["source"]).stem.replace(" ", "_"),
        )
        folder = filedialog.askdirectory(title="Choose a parent folder for this test run")
        if not folder:
            return

        run_dir = Path(folder) / default_name
        run_dir.mkdir(parents=True, exist_ok=False)

        if vis is not None:
            cv2.imwrite(str(run_dir / "annotated.jpg"), vis)

        (run_dir / "detections.json").write_text(json.dumps(
            {"run": meta, "verdict": summary, "detections": detections}, indent=2))

        (run_dir / "report.txt").write_text(self._format_report(meta, summary))

        self._log("Saved test run -> %s" % run_dir)
        messagebox.showinfo("Saved", "Test run saved to:\n%s" % run_dir)

    def _format_report(self, meta, summary):
        lines = [
            "BIN FULLNESS DETECTOR - TEST RUN REPORT",
            "=" * 44,
            "Run at      : %s" % meta["run_at"],
            "Model       : %s" % meta["model"],
            "Classes     : %s" % ", ".join(meta["classes"]),
            "Source      : %s" % meta["source"],
            "Confidence threshold : %.2f" % meta["confidence_threshold"],
            "IoU threshold        : %.2f" % meta["iou_threshold"],
            "Video stride         : %d" % meta["video_stride"],
            "Min frames required  : %d" % meta["min_frames"],
            "",
            "VERDICT",
            "-" * 44,
            self._format_verdict(summary),
        ]
        return "\n".join(lines) + "\n"


def main():
    root = tk.Tk()
    DetectGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
