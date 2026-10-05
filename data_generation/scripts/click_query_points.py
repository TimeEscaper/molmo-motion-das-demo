#!/usr/bin/env python3
"""Manual replacement for pipeline Stage 1 (grounding): click query points on a frame.

Opens the first frame of each video (or an image) in an OpenCV window. You click the
pixels to track, and they are saved in the same format Stage 1 writes, so the pipeline
can continue from Stage 2:

    <out_dir>/<video_id>/query_points/<video_id>_manual_obj<k>_f<frame>.npz
        query_points  (N, 3) int32  [frame_idx, x, y]   in source-video pixels
        dim           (2,)   int32  [H, W]              source-video resolution
    <out_dir>/<video_id>/query_points/<video_id>_manual.json   (same points, readable)

Point <out_dir> at `<work_dir>/grounding` (or copy it there on the server), then run
    python run_pipeline.py --tasks ... --config configs/robot.yaml --work_dir <work_dir> --start_stage 2
Each object becomes its own NPZ, which Stage 5 uses to filter/smooth objects separately.

With --episodes, the inputs are episode folders (or folders of them) as written by
scripts/data/extract_sharerobot.py, and the points go to <episode>/query_points/, where
`run_pipeline.py --episodes ... --query_points manual` picks them up.

Only needs numpy + opencv-python (not the -headless build, which has no GUI).

Controls:
    left click       add point to the current object
    right click      remove the nearest point
    u                undo last point
    n                start a new object
    c                clear all points
    s / Enter        save and go to the next video
    k                skip this video (nothing saved)
    q / Esc          quit (current video is not saved)

Usage:
    python click_query_points.py video1.mp4 video2.mp4 --out_dir runs/my_run/grounding
    python click_query_points.py videos_dir/ --out_dir runs/my_run/grounding
    python click_query_points.py episode/frame_0.png --video_id bridge_4899 --out_dir ...
    python click_query_points.py --episodes ../result/sharerobot_raw
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # BGR


def read_frame(path: Path, frame_idx: int) -> np.ndarray:
    if path.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(path))
    else:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        cap.release()
        img = img if ok else None
    if img is None:
        raise RuntimeError(f"cannot read frame {frame_idx} from {path}")
    return img


class Clicker:
    def __init__(self, img: np.ndarray, title: str, max_w: int, max_h: int):
        self.img = img
        self.title = title
        h, w = img.shape[:2]
        self.scale = min(1.0, max_w / w, max_h / h)
        self.objects = [[]]  # list of objects, each a list of (x, y) in source pixels

    def to_src(self, x, y):
        h, w = self.img.shape[:2]
        return (min(w - 1, max(0, int(round(x / self.scale)))),
                min(h - 1, max(0, int(round(y / self.scale)))))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.objects[-1].append(self.to_src(x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            sx, sy = self.to_src(x, y)
            best = None
            for oi, pts in enumerate(self.objects):
                for pi, (px, py) in enumerate(pts):
                    d = (px - sx) ** 2 + (py - sy) ** 2
                    if best is None or d < best[0]:
                        best = (d, oi, pi)
            if best is not None:
                del self.objects[best[1]][best[2]]

    def render(self) -> np.ndarray:
        h, w = self.img.shape[:2]
        vis = cv2.resize(self.img, (int(w * self.scale), int(h * self.scale)),
                         interpolation=cv2.INTER_AREA) if self.scale < 1 else self.img.copy()
        for oi, pts in enumerate(self.objects):
            color = COLORS[oi % len(COLORS)]
            for pi, (px, py) in enumerate(pts):
                p = (int(px * self.scale), int(py * self.scale))
                cv2.circle(vis, p, 4, color, -1)
                cv2.circle(vis, p, 5, (255, 255, 255), 1)
                cv2.putText(vis, f"{oi}.{pi}", (p[0] + 6, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        counts = " ".join(f"obj{i}:{len(p)}" for i, p in enumerate(self.objects))
        status = f"{self.title} | {counts} | n=new obj u=undo s=save k=skip q=quit"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(vis, status, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def run(self) -> str:
        cv2.namedWindow("click_query_points", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("click_query_points", self.on_mouse)
        while True:
            cv2.imshow("click_query_points", self.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("s"), 13, 10):
                return "save"
            if key == ord("k"):
                return "skip"
            if key in (ord("q"), 27):
                return "quit"
            if key == ord("n") and self.objects[-1]:
                self.objects.append([])
            elif key == ord("u"):
                for pts in reversed(self.objects):
                    if pts:
                        pts.pop()
                        break
                while len(self.objects) > 1 and not self.objects[-1] and not self.objects[-2]:
                    self.objects.pop()
            elif key == ord("c"):
                self.objects = [[]]


def save(objects, img_shape, frame_idx, video_id, source, qp_dir: Path):
    qp_dir.mkdir(parents=True, exist_ok=True)
    for old in qp_dir.glob(f"{video_id}_manual_obj*_f*.npz"):
        old.unlink()
    h, w = img_shape[:2]
    objects = [pts for pts in objects if pts]
    for oi, pts in enumerate(objects):
        q = np.array([[frame_idx, x, y] for x, y in pts], dtype=np.int32)
        np.savez_compressed(qp_dir / f"{video_id}_manual_obj{oi}_f{frame_idx}.npz",
                            query_points=q, dim=np.asarray([h, w], dtype=np.int32))
    meta = {"video_id": video_id, "source": str(source), "frame_idx": frame_idx,
            "dim": [h, w], "objects": {f"obj{i}": [list(p) for p in pts]
                                       for i, pts in enumerate(objects)}}
    (qp_dir / f"{video_id}_manual.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved {sum(map(len, objects))} points in {len(objects)} object(s) -> {qp_dir}")


def collect_inputs(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS)
        else:
            out.append(p)
    return out


def collect_episodes(paths):
    """(video, video_id, query_points dir) of episode folders, given directly or as parents."""
    out = []
    for root in map(Path, paths):
        dirs = [root] if (root / "episode.json").exists() else sorted(
            d for d in root.iterdir() if (d / "episode.json").exists())
        for d in dirs:
            meta = json.loads((d / "episode.json").read_text())
            video = d / meta["video"] if "video" in meta else next(d.glob("*.mp4"))
            out.append((video, meta.get("video_id", video.stem), d / "query_points"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="videos, images, or directories of them")
    ap.add_argument("--out_dir", default=None, help="grounding dir, i.e. <work_dir>/grounding")
    ap.add_argument("--episodes", action="store_true",
                    help="inputs are episode folders; save to <episode>/query_points/")
    ap.add_argument("--frame", type=int, default=0, help="frame index to annotate (default 0)")
    ap.add_argument("--video_id", default=None, help="override video_id (single input only; default: file stem)")
    ap.add_argument("--overwrite", action="store_true", help="re-annotate videos that already have points")
    ap.add_argument("--max_w", type=int, default=1400, help="max window width (image is downscaled to fit)")
    ap.add_argument("--max_h", type=int, default=850, help="max window height")
    args = ap.parse_args()

    if args.episodes == bool(args.out_dir):
        ap.error("pass exactly one of --out_dir / --episodes")
    if args.episodes:
        inputs = collect_episodes(args.inputs)
    else:
        inputs = [(p, args.video_id or p.stem, Path(args.out_dir) / (args.video_id or p.stem) / "query_points")
                  for p in collect_inputs(args.inputs)]
    if args.video_id and (args.episodes or len(inputs) != 1):
        ap.error("--video_id requires exactly one input (and no --episodes)")

    for i, (path, vid, qp_dir) in enumerate(inputs):
        existing = list(qp_dir.glob(f"{vid}_*_f*.npz"))
        if existing and not args.overwrite:
            print(f"[{i + 1}/{len(inputs)}] {vid}: already annotated, skipping (--overwrite to redo)")
            continue
        try:
            img = read_frame(path, args.frame)
        except RuntimeError as e:
            print(f"[{i + 1}/{len(inputs)}] {vid}: {e}")
            continue
        print(f"[{i + 1}/{len(inputs)}] {vid}  ({img.shape[1]}x{img.shape[0]})")
        clicker = Clicker(img, f"[{i + 1}/{len(inputs)}] {vid}", args.max_w, args.max_h)
        action = clicker.run()
        if action == "quit":
            break
        if action == "save":
            if any(clicker.objects):
                save(clicker.objects, img.shape, args.frame, vid, path, qp_dir)
            else:
                print("  no points clicked, nothing saved")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
