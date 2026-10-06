"""Convert MolmoSpaces clips into the bundled-example layout of `examples/01_quickstart.py`.

Each converted clip becomes `<output_root>/molmospaces_<video>[_stride<N>]/` (the
suffix only when `--stride N` > 1) with the same
files as `examples/data/davis_bmx_trees/` plus the ground-truth future:

    frame_t-2.jpg, frame_t-1.jpg, frame_t+0.jpg   H history frames (earliest → t0)
    points_2d_at_t0.pt       (P, 2)     pixel xy of the query points at t0
    points_3d_history.pt     (H, P, 3)  camera-frame-at-t0 XYZ, meters
    intrinsics_K.pt          (3, 3)
    caption.txt, meta.json
    clip.mp4                 history + future frames, at fps = 15 / stride
    gt_future_3d.pt          (P, F, 3)  camera-frame-at-t0 XYZ (same frame as `out.future_3d`)
    gt_future_vis.pt         (P, F)     bool

Only static cameras are used (every MolmoSpaces camera except the wrist ones).
If the chosen object has several motion clips, the last one (latest start
across all objects) is used, and t0 is placed `--offset` raw frames after its
start so the object is already moving in the history window.

Usage:
    python scripts/data/sample_molmospaces.py \\
        --video pick_place_2cam_randomized__house_102__00000000__exo_camera_1
    python scripts/data/sample_molmospaces.py --num_videos 5 --seed 0

Then run the model on it:
    python scripts/run_molmo_motion.py --input result/molmo_motion_input/molmospaces/molmospaces_<video>[_stride<N>]
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

# Repo uses a src layout (`src/molmo_motion`). This script lives in
# scripts/data/, so the repo root is two levels up. Load `.env` before any
# MOLMO_MOTION_1M_ROOT lookup.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

# Track NPZs hold numpy-2 pickles and the venv has numpy 1.26: alias numpy._core (run the
# module file, not `import molmo_motion...`, which would load the whole package).
runpy.run_path(str(REPO_ROOT / "src" / "molmo_motion" / "numpy_compat.py"))

DEFAULT_DATA_ROOT = Path(os.environ.get(
    "MOLMO_MOTION_1M_ROOT", "/mnt/vol1/shared/datasets/molmo-motion-1m")) / "molmospaces"


class SkipVideo(Exception):
    pass


def stride_suffix(stride: int) -> str:
    """Output-dir suffix for temporally strided examples ('' at the native rate)."""
    return f"_stride{stride}" if stride != 1 else ""


def is_static_camera_name(cam: str) -> bool:
    return "wrist" not in cam


def load_entries(data_root: Path) -> dict[str, list[dict]]:
    with open(data_root / "annotations" / "molmospaces_split.json") as f:
        split = json.load(f)
    return {"train": split["train"], "test": split["test"]}


def last_clip(entry: dict) -> tuple[str, int, int]:
    """(obj, start, end) of the clip with the latest start across all objects; end inclusive."""
    clips = [(s, e, obj) for obj, ranges in entry["clips_by_object"].items() for s, e in ranges]
    s, e, obj = max(clips)
    return obj, int(s), int(e)


def read_frames(mp4_path: Path, first: int, last: int) -> np.ndarray:
    """Read raw frames [first, last] inclusive as (N, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise SkipVideo(f"cannot open {mp4_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for _ in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != last - first + 1:
        raise SkipVideo(f"read {len(frames)} frames, expected {last - first + 1} from {mp4_path}")
    return np.stack(frames)


def convert(entry: dict, data_root: Path, output_root: Path, *, history: int, future: int,
            num_points: int, stride: int, offset: int) -> Path:
    vid = entry["file"]
    if not is_static_camera_name(entry["cam"]):
        raise SkipVideo(f"camera '{entry['cam']}' is not static")

    obj, clip_s, clip_e = last_clip(entry)
    num_frames = int(entry["num_frames"])
    fps = float(entry.get("fps", 15))

    t0 = clip_s + offset
    hist_idx = [t0 - k * stride for k in range(history - 1, -1, -1)]
    if hist_idx[0] < 0:
        raise SkipVideo(f"history starts before frame 0 (t0={t0})")
    if t0 > clip_e:
        raise SkipVideo(f"t0={t0} is past the clip end {clip_e} (clip {clip_s}-{clip_e}, offset {offset})")
    fut_idx = [t0 + k * stride for k in range(1, future + 1) if t0 + k * stride <= num_frames - 1]
    if not fut_idx:
        raise SkipVideo(f"no future frames after t0={t0} (num_frames={num_frames})")

    with np.load(data_root / "tracks" / f"{vid}_3d.npz", allow_pickle=True) as f:
        pts_3d = f["points_3d"].item()[obj].astype(np.float32)   # (K, T, 3) world
        vis_3d = f["visibility"].item()[obj].astype(bool)        # (K, T)
    with np.load(data_root / "tracks" / f"{vid}_2d.npz", allow_pickle=True) as f:
        tracks_2d = f["tracks"].item()[obj].astype(np.float32)   # (T, K, 2) pixels
    with np.load(data_root / "camera" / f"{vid}.npz") as f:
        cam_poses = f["cam_poses"].astype(np.float64)            # (T, 4, 4) c2w
        K = f["intrinsics"].astype(np.float32)                   # (3, 3)
    if not np.allclose(cam_poses, cam_poses[0], atol=1e-6):
        raise SkipVideo(f"camera '{entry['cam']}' moves (cam_poses not constant)")

    # Query points: visible with finite coords over the whole history, uniformly
    # spaced over the candidate pool (same convention as the eval pipeline).
    finite = np.isfinite(pts_3d[:, hist_idx]).all(axis=(1, 2)) & np.isfinite(tracks_2d[t0]).all(axis=1)
    cand = np.where(vis_3d[:, hist_idx].all(axis=1) & finite)[0]
    if len(cand) < num_points:
        raise SkipVideo(f"only {len(cand)} points visible over the history (need {num_points})")
    chosen = cand[[int(i * len(cand) / num_points) for i in range(num_points)]]

    w2c = np.linalg.inv(cam_poses[t0])
    R, tvec = w2c[:3, :3].astype(np.float32), w2c[:3, 3].astype(np.float32)
    to_cam = lambda x: x @ R.T + tvec  # noqa: E731

    points_3d_history = to_cam(pts_3d[chosen][:, hist_idx]).transpose(1, 0, 2)   # (H, P, 3)
    gt_future_3d = to_cam(np.nan_to_num(pts_3d[chosen][:, fut_idx]))            # (P, F, 3)
    gt_future_vis = vis_3d[chosen][:, fut_idx] & np.isfinite(pts_3d[chosen][:, fut_idx]).all(-1)
    points_2d_at_t0 = tracks_2d[t0, chosen]                                      # (P, 2)

    raw = read_frames(data_root / "videos" / f"{vid}.mp4", hist_idx[0], fut_idx[-1])
    clip_frames = raw[::stride]                     # hist_idx + fut_idx, in order

    out_dir = output_root / f"molmospaces_{vid}{stride_suffix(stride)}"
    meta = save_example(
        out_dir, clip_frames=clip_frames, fps=fps / stride, points_2d_at_t0=points_2d_at_t0,
        points_3d_history=points_3d_history, K=K, gt_future_3d=gt_future_3d,
        gt_future_vis=gt_future_vis, caption=entry["caption"],
        meta={
            "dataset": "molmospaces",
            "video": vid,
            "camera": entry["cam"],
            "obj": obj,
            "t0_absolute": t0,
            "history_frame_indices": hist_idx,
            "future_frame_indices": fut_idx,
            "point_indices": chosen.tolist(),
            "clip_range": [clip_s, clip_e],
            "offset": offset,
            "stride": stride,
            "native_fps": fps,
            "effective_fps": fps / stride,
        },
    )
    print(f"[ok] {out_dir}  obj={obj} clip=[{clip_s},{clip_e}] t0={t0} F={len(fut_idx)} "
          f"history_motion={meta['history_motion_m'] * 100:.1f}cm "
          f"future_motion={meta['future_motion_m'] * 100:.1f}cm")
    return out_dir


def save_example(out_dir: Path, *, clip_frames: np.ndarray, fps: float, points_2d_at_t0: np.ndarray,
                 points_3d_history: np.ndarray, K: np.ndarray, gt_future_3d: np.ndarray,
                 gt_future_vis: np.ndarray, caption: str, meta: dict) -> dict:
    """Write one example in the `examples/01_quickstart.py` layout.

    `clip_frames` holds the H history frames followed by the future frames (one
    per model step); `meta` must carry `t0_absolute`, `history_frame_indices`
    and `future_frame_indices` and is completed with the common fields.
    """
    history, num_points = points_3d_history.shape[:2]
    hist_frames = clip_frames[:history]
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, frame in zip(range(-(history - 1), 1), hist_frames):
        Image.fromarray(frame).save(out_dir / f"frame_t{k:+d}.jpg", quality=95)
    torch.save(torch.from_numpy(np.ascontiguousarray(points_2d_at_t0, dtype=np.float32)),
               out_dir / "points_2d_at_t0.pt")
    torch.save(torch.from_numpy(np.ascontiguousarray(points_3d_history, dtype=np.float32)),
               out_dir / "points_3d_history.pt")
    torch.save(torch.from_numpy(np.ascontiguousarray(K, dtype=np.float32)), out_dir / "intrinsics_K.pt")
    torch.save(torch.from_numpy(np.ascontiguousarray(gt_future_3d, dtype=np.float32)),
               out_dir / "gt_future_3d.pt")
    torch.save(torch.from_numpy(np.ascontiguousarray(gt_future_vis, dtype=bool)), out_dir / "gt_future_vis.pt")
    (out_dir / "caption.txt").write_text(caption.strip() + "\n")
    imageio.mimsave(out_dir / "clip.mp4", list(clip_frames), fps=fps, codec="libx264", macro_block_size=1)

    img_h, img_w = hist_frames.shape[1:3]
    full_meta = {
        "id": out_dir.name,
        **meta,
        "n_points": int(num_points),
        "history_size": int(history),
        "future_size": len(meta["future_frame_indices"]),
        "image_size_wh": [int(img_w), int(img_h)],
        "has_intrinsics": True,
        "caption": caption.strip(),
        "clip_mp4_frame_indices": list(meta["history_frame_indices"]) + list(meta["future_frame_indices"]),
        "history_motion_m": float(np.linalg.norm(points_3d_history[-1] - points_3d_history[0], axis=-1).mean()),
        "future_motion_m": float(np.linalg.norm(gt_future_3d[:, -1] - points_3d_history[-1], axis=-1).mean()),
    }
    (out_dir / "meta.json").write_text(json.dumps(full_meta, indent=2) + "\n")
    return full_meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Video name `{scenario}__{house}__{episode}__{cam}` (train or test split).")
    src.add_argument("--num_videos", type=int, help="Randomly sample this many static-camera test videos.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--output_root", type=Path, default=REPO_ROOT / "result" / "molmo_motion_input" / "molmospaces")
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--future", type=int, default=32)
    ap.add_argument("--num_points", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1,
                    help="Temporal subsampling of the 15 fps source (1 = native 15 fps, which "
                         "matches the human-video data the released models were trained on).")
    ap.add_argument("--offset", type=int, default=10,
                    help="t0 = start of the last motion clip + offset (raw frames).")
    args = ap.parse_args()

    splits = load_entries(args.data_root)
    kwargs = dict(history=args.history, future=args.future, num_points=args.num_points,
                  stride=args.stride, offset=args.offset)

    if args.video is not None:
        name = args.video.removesuffix(".mp4")
        entry = next((e for es in splits.values() for e in es if e["file"] == name), None)
        if entry is None:
            raise SystemExit(f"video {name!r} not found in molmospaces_split.json")
        try:
            convert(entry, args.data_root, args.output_root, **kwargs)
        except SkipVideo as e:
            raise SystemExit(f"cannot convert {name}: {e}")
        return

    candidates = [e for e in splits["test"] if is_static_camera_name(e["cam"])]
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
