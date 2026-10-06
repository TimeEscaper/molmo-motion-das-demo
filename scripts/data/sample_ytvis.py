"""Convert YT-VIS (MolmoMotion-1M) clips into the bundled-example layout of `examples/01_quickstart.py`.

Writes `<output_root>/ytvis_<video>[_stride<N>]/` (the suffix only when
`--stride N` > 1) with the same files as
`scripts/data/sample_molmospaces.py` (history frames, query points, 3D
history, intrinsics, caption, meta.json, clip.mp4, gt_future_3d.pt,
gt_future_vis.pt), so `--side-by-side` in scripts/run_molmo_motion.py works on it too.

YT-VIS specifics:
  * Cameras are hand-held (moving), so every 3D point is expressed in the
    camera frame at t0 (`inv(pose[t0])`); intrinsics come from frame t0.
  * Clips are short (18-50 frames) at their native per-clip fps (6/8/12). The
    training loader uses them without temporal striding, so neither do we by
    default (`--stride 1`); the future is truncated at the clip end when the
    clip is shorter than H + F.
  * Each object has one motion clip, usually spanning the whole video. With
    several objects, the one that moves most over the future window is used
    (override with `--obj`).
  * Objects often enter the frame during the first frames, so t0 starts at
    `clip start + --offset` and advances until enough query points are
    visible over the whole history.
  * Image size is taken from the video: `dim` in the 2D NPZ is always
    [480, 854] even for narrower videos, while tracks and intrinsics are in
    the video's actual pixel coordinates.

Usage:
    python scripts/data/sample_ytvis.py --video 0043f083b5
    python scripts/data/sample_ytvis.py --num_videos 5 --seed 0

Then run the model on it:
    python scripts/run_molmo_motion.py --input result/molmo_motion_input/ytvis/ytvis_<video>[_stride<N>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import numpy as np

# Repo uses a src layout (`src/molmo_motion`). This script lives in
# scripts/data/, so the repo root is two levels up. Load `.env` before any
# MOLMO_MOTION_1M_ROOT lookup.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

from sample_molmospaces import SkipVideo, read_frames, save_example, stride_suffix

DEFAULT_DATA_ROOT = Path(os.environ.get(
    "MOLMO_MOTION_1M_ROOT", "/mnt/vol1/shared/datasets/molmo-motion-1m")) / "ytvis"


def load_entries(data_root: Path) -> dict[str, list[dict]]:
    with open(data_root / "annotations" / "ytvis_split.json") as f:
        split = json.load(f)
    return {"train": split["train"], "test": split["test"]}


def prepare_object(obj: str, ranges: list, pts_3d: np.ndarray, vis_3d: np.ndarray,
                   tracks_2d: np.ndarray, poses: np.ndarray, intrinsics: np.ndarray,
                   img_wh: tuple[int, int], *, history: int, future: int, num_points: int,
                   offset: int, min_future: int, stride: int) -> dict:
    """Pick t0 / query points for one object and express everything in the camera frame at t0.

    t0 starts at `clip start + offset` and advances until at least `num_points`
    points are visible over the whole history (objects often enter the frame
    during the first frames) while `min_future` future steps remain. History
    and future frames are `stride` raw frames apart.

    pts_3d (N, T, 3) world, vis_3d (N, T), tracks_2d (T, N, 2) pixels,
    poses (T, 4, 4) c2w, intrinsics (T, 4) [fx, fy, cx, cy].
    """
    clip_s, clip_e = (int(x) for x in max(ranges))          # last clip; end inclusive
    clip_e = min(clip_e, pts_3d.shape[1] - 1)
    span = (history - 1) * stride
    if clip_s + offset - span < 0:
        raise SkipVideo(f"history starts before frame 0; use --offset >= {span}")
    last_t0 = min(clip_e - min_future * stride, clip_e - stride)
    if clip_s + offset > last_t0:
        raise SkipVideo(f"clip [{clip_s},{clip_e}] too short for offset {offset} + {min_future} future "
                        f"steps at stride {stride}")

    best_n = 0
    for t0 in range(clip_s + offset, last_t0 + 1):
        # Query points: visible with finite coords over the whole history and
        # inside the image at t0.
        hist_idx = list(range(t0 - span, t0 + 1, stride))
        uv0 = tracks_2d[t0]
        in_image = ((uv0 >= 0) & (uv0 < np.array(img_wh))).all(axis=1)
        finite = np.isfinite(pts_3d[:, hist_idx]).all(axis=(1, 2)) & np.isfinite(uv0).all(axis=1)
        cand = np.where(vis_3d[:, hist_idx].all(axis=1) & finite & in_image)[0]
        best_n = max(best_n, len(cand))
        if len(cand) >= num_points:
            break
    else:
        raise SkipVideo(f"at most {best_n} points visible over any history window with "
                        f">= {min_future} future frames left (need {num_points})")
    chosen = cand[[int(i * len(cand) / num_points) for i in range(num_points)]]
    fut_idx = list(range(t0 + stride, min(clip_e, t0 + future * stride) + 1, stride))

    w2c = np.linalg.inv(poses[t0].astype(np.float64))
    R, tvec = w2c[:3, :3].astype(np.float32), w2c[:3, 3].astype(np.float32)
    to_cam = lambda x: x @ R.T + tvec  # noqa: E731
    fx, fy, cx, cy = intrinsics[t0]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    points_3d_history = to_cam(pts_3d[chosen][:, hist_idx]).transpose(1, 0, 2)   # (H, P, 3)
    fut_raw = pts_3d[chosen][:, fut_idx]
    gt_future_3d = to_cam(np.nan_to_num(fut_raw))                                # (P, F, 3)
    gt_future_vis = vis_3d[chosen][:, fut_idx] & np.isfinite(fut_raw).all(-1)
    motion = float(np.linalg.norm(gt_future_3d[:, -1] - points_3d_history[-1], axis=-1).mean())
    return {
        "obj": obj, "clip_range": [clip_s, clip_e], "t0": t0, "hist_idx": hist_idx,
        "fut_idx": fut_idx, "chosen": chosen, "points_2d_at_t0": uv0[chosen],
        "points_3d_history": points_3d_history, "gt_future_3d": gt_future_3d,
        "gt_future_vis": gt_future_vis, "K": K, "motion": motion,
    }


def convert(entry: dict, data_root: Path, output_root: Path, *, history: int, future: int,
            num_points: int, offset: int, min_future: int, stride: int = 1,
            obj: str | None = None) -> Path:
    vid = entry["file"]
    num_frames = int(entry["num_frames"])
    fps = float(entry["fps"])
    mp4 = data_root / "videos" / f"{vid}.mp4"
    if not mp4.exists():
        raise SkipVideo(f"{mp4} missing; reconstruct videos with {data_root}/reconstruct_videos.py")
    if obj is not None and obj not in entry["clips_by_object"]:
        raise SkipVideo(f"object {obj!r} not in {sorted(entry['clips_by_object'])}")

    with np.load(data_root / "tracks" / f"{vid}_3d.npz", allow_pickle=True) as f:
        pts_dict = f["points_3d"].item()                     # {obj: (N, T, 3)} world
        vis_dict = f["visibility"].item()                    # {obj: (N, T)}
    with np.load(data_root / "tracks" / f"{vid}_2d.npz", allow_pickle=True) as f:
        tracks_dict = f["tracks"].item()                     # {obj: (T, N, 2)} pixels
    with np.load(data_root / "camera" / "pose" / f"{vid}.npz") as f:
        poses = f["data"]                                    # (T, 4, 4) c2w
    with np.load(data_root / "camera" / "intrinsics" / f"{vid}.npz") as f:
        intrinsics = f["data"]                               # (T, 4) [fx, fy, cx, cy]

    frames = read_frames(mp4, 0, num_frames - 1)
    img_wh = (frames.shape[2], frames.shape[1])

    options, reasons = [], []
    for o in ([obj] if obj is not None else sorted(entry["clips_by_object"])):
        if o not in pts_dict:
            reasons.append(f"{o}: missing from tracks NPZ")
            continue
        pts = pts_dict[o].astype(np.float32)
        try:
            options.append(prepare_object(
                o, entry["clips_by_object"][o], pts, vis_dict[o].reshape(pts.shape[:2]).astype(bool),
                tracks_dict[o].astype(np.float32), poses, intrinsics, img_wh,
                history=history, future=future, num_points=num_points, offset=offset,
                min_future=min_future, stride=stride))
        except SkipVideo as e:
            reasons.append(f"{o}: {e}")
    if not options:
        raise SkipVideo("; ".join(reasons))
    best = max(options, key=lambda x: x["motion"])

    out_dir = output_root / f"ytvis_{vid}{stride_suffix(stride)}"
    meta = save_example(
        out_dir, clip_frames=frames[best["hist_idx"] + best["fut_idx"]], fps=fps / stride,
        points_2d_at_t0=best["points_2d_at_t0"], points_3d_history=best["points_3d_history"],
        K=best["K"], gt_future_3d=best["gt_future_3d"], gt_future_vis=best["gt_future_vis"],
        caption=entry["caption"],
        meta={
            "dataset": "ytvis",
            "video": vid,
            "obj": best["obj"],
            "objects_available": sorted(entry["clips_by_object"]),
            "t0_absolute": best["t0"],
            "history_frame_indices": best["hist_idx"],
            "future_frame_indices": best["fut_idx"],
            "point_indices": best["chosen"].tolist(),
            "clip_range": best["clip_range"],
            "offset": offset,
            "stride": stride,
            "native_fps": fps,
            "effective_fps": fps / stride,
        },
    )
    print(f"[ok] {out_dir}  obj={best['obj']} (of {len(entry['clips_by_object'])}) "
          f"clip={best['clip_range']} t0={best['t0']} F={len(best['fut_idx'])} fps={fps / stride:g} "
          f"history_motion={meta['history_motion_m'] * 100:.1f}cm "
          f"future_motion={meta['future_motion_m'] * 100:.1f}cm")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="10-hex YT-VIS video id (train or test split).")
    src.add_argument("--num_videos", type=int, help="Randomly sample this many test videos.")
    ap.add_argument("--obj", default=None,
                    help="Object to use with --video (e.g. person_0); default = the one that moves most.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--output_root", type=Path, default=REPO_ROOT / "result" / "molmo_motion_input" / "ytvis")
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--future", type=int, default=32)
    ap.add_argument("--num_points", type=int, default=8)
    ap.add_argument("--offset", type=int, default=2,
                    help="Earliest t0 = clip start + offset (frames); t0 advances further while "
                         "fewer than --num_points points are visible over the history. Must be "
                         ">= (history - 1) * stride; YT-VIS clips are short, so every extra frame "
                         "of offset shortens the future.")
    ap.add_argument("--min_future", type=int, default=8,
                    help="Skip objects/videos with fewer future frames (model steps) than this inside the clip.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Temporal subsampling of the native per-clip fps (1 = native, as in training).")
    args = ap.parse_args()
    if args.obj is not None and args.video is None:
        ap.error("--obj requires --video")
    if args.stride < 1:
        ap.error("--stride must be >= 1")

    splits = load_entries(args.data_root)
    kwargs = dict(history=args.history, future=args.future, num_points=args.num_points,
                  offset=args.offset, min_future=args.min_future, stride=args.stride)

    if args.video is not None:
        entry = next((e for es in splits.values() for e in es if e["file"] == args.video), None)
        if entry is None:
            raise SystemExit(f"video {args.video!r} not found in ytvis_split.json")
        try:
            convert(entry, args.data_root, args.output_root, obj=args.obj, **kwargs)
        except SkipVideo as e:
            raise SystemExit(f"cannot convert {args.video}: {e}")
        return

    candidates = splits["test"]
    order = np.random.RandomState(args.seed).permutation(len(candidates))
    n_done = 0
    for i in order:
        if n_done >= args.num_videos:
            break
        try:
            convert(candidates[i], args.data_root, args.output_root, **kwargs)
            n_done += 1
        except SkipVideo as e:
            print(f"[skip] {candidates[i]['file']}: {e}")
    print(f"converted {n_done}/{args.num_videos} videos into {args.output_root}")


if __name__ == "__main__":
    main()
