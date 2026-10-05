"""Convert ShareRobot episodes annotated by the data-generation pipeline into the bundled-example layout of `examples/01_quickstart.py`.

Writes `<output_root>/sharerobot_<video>[_stride<N>]/` (the suffix only when
`--stride N` > 1) with the same files as `scripts/data/sample_molmospaces.py`
(history frames, query points, 3D history, intrinsics, caption, meta.json,
clip.mp4, gt_future_3d.pt, gt_future_vis.pt), plus the episode's init frame:

    frame_init.jpg           the first frame of the episode (`--init_frame`), before the
                             gripper reaches / occludes the object
    points_2d_at_init.pt     (P, 2)     pixel xy of the same query points at the init frame
    points_3d_at_init.pt     (1, P, 3)  their XYZ in the camera frame at the init frame
                             (a drop-in `points_3d_history` for the H=1 model)
    intrinsics_K_init.pt     (3, 3)

so the H=1 model can be run from the init frame and scored on the same future
(`gt_future_3d.pt` stays the future of t0, in the camera frame at t0; with a
fixed camera both frames coincide).

Input is a work dir of `data_generation/run_pipeline.py --episodes ...`, one
folder per video:
    <video>/meta.json                       action (episode goal) + episode.json
    <video>/video.mp4                       the video the tracker saw
    <video>/final_tracks/<video>_{3d,2d}.npz  filtered world-frame 3D + 2D tracks
    <video>/camera/{intrinsics,pose}.npz    per-frame K + camera-to-world
    <video>/clips.json                      motion clips

ShareRobot specifics:
  * Episodes are 30 frames subsampled from the OXE source (~0.5-15 fps; bridge
    ~4 fps), kept at their native rate by default (`--stride 1`); the future is
    truncated at the last frame.
  * t0 = start of the object's (last) motion clip + `--offset`, advanced until
    `--num_points` points are visible over the whole history and at the init
    frame; override with `--t0`.
  * The caption is the episode goal (`--caption goal`) or the sub-step active at
    t0 (`--caption step`); both are stored in meta.json.

Usage:
    python scripts/data/sample_sharerobot.py --video bridge_3715
    python scripts/data/sample_sharerobot.py --num_videos 2 --seed 0
    python scripts/data/sample_sharerobot.py --video bridge_3715 \\
        --data_root result/datagen/sharerobot_raw_da3 --tag da3

Then run the model on it:
    python scripts/run_molmo_motion.py --input result/molmo_motion_input/sharerobot/sharerobot_<video>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import torch
from PIL import Image

# Repo uses a src layout (`src/molmo_motion`). This script lives in
# scripts/data/, so the repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

from sample_molmospaces import SkipVideo, read_frames, save_example, stride_suffix

DEFAULT_DATA_ROOT = REPO_ROOT / "result" / "datagen" / "sharerobot_raw"


def list_videos(data_root: Path) -> list[str]:
    """Video folders of a pipeline work dir that have final tracks."""
    return sorted(d.name for d in data_root.iterdir()
                  if (d / "final_tracks" / f"{d.name}_3d.npz").exists())


def step_at(episode: dict, frame: int) -> dict | None:
    for st in episode.get("steps", []):
        if st["start"] is not None and st["end"] is not None and st["start"] <= frame < st["end"]:
            return st
    return None


def camera_K(intr, t: int, img_wh: tuple[int, int]) -> np.ndarray:
    """K of frame t, rescaled from the camera's image size to the video's."""
    K = intr["K"][t].astype(np.float32).copy()
    cam_w, cam_h = (int(v) for v in intr["image_size_wh"])
    K[0] *= img_wh[0] / cam_w
    K[1] *= img_wh[1] / cam_h
    return K


def convert(vid: str, data_root: Path, output_root: Path, *, history: int, future: int,
            num_points: int, offset: int, min_future: int, t0_fixed: int | None, stride: int,
            init_frame: int, caption_src: str, tag: str | None) -> Path:
    vdir = data_root / vid
    meta_in = json.loads((vdir / "meta.json").read_text())
    clips = json.loads((vdir / "clips.json").read_text()) if (vdir / "clips.json").exists() else []
    ranges = clips[0]["clips_by_object"].get("obj0") or next(iter(clips[0]["clips_by_object"].values()), None) \
        if clips else None
    if not ranges:
        raise SkipVideo("no motion clip found by stage 6")
    clip_s, clip_e = (int(x) for x in max(ranges))          # last clip; end inclusive
    fps = float(clips[0]["fps"])

    with np.load(vdir / "final_tracks" / f"{vid}_3d.npz", allow_pickle=True) as f:
        if f["points_3d"].dtype == object:
            raise SkipVideo("nested per-object final tracks are not supported")
        pts_3d = f["points_3d"].astype(np.float32)          # (N, T, 3) world
        vis_3d = f["visibility"].reshape(pts_3d.shape[:2]).astype(bool)
    with np.load(vdir / "final_tracks" / f"{vid}_2d.npz", allow_pickle=True) as f:
        tracks_2d = f["tracks"].astype(np.float32)          # (T, N, 2) video pixels
    with np.load(vdir / "camera" / "pose.npz") as f:
        poses = f["data"].astype(np.float64)                # (T, 4, 4) camera-to-world
    intr = dict(np.load(vdir / "camera" / "intrinsics.npz"))

    video = next(vdir.glob("video.*"))
    T = min(pts_3d.shape[1], len(tracks_2d), len(poses), len(intr["K"]))
    frames = read_frames(video, 0, T - 1)
    img_wh = (frames.shape[2], frames.shape[1])
    if not 0 <= init_frame < T:
        raise SkipVideo(f"init frame {init_frame} outside the episode (T={T})")

    def usable(t: int) -> np.ndarray:                       # visible, finite, inside the image
        uv = tracks_2d[t]
        return (vis_3d[:, t] & np.isfinite(pts_3d[:, t]).all(-1) & np.isfinite(uv).all(-1)
                & ((uv >= 0) & (uv < np.array(img_wh))).all(-1))

    # t0: query points visible over the whole history (and at the init frame).
    span = (history - 1) * stride                           # raw frames covered by the history
    first = t0_fixed if t0_fixed is not None else max(clip_s + offset, span)
    last = first if t0_fixed is not None else T - 1 - min_future * stride
    if first < span or first > T - 1 - stride:
        raise SkipVideo(f"t0={first} leaves no room for {history} history frames and a future "
                        f"(T={T}, stride={stride})")
    at_init, best_n = usable(init_frame), 0
    for t0 in range(first, last + 1):
        hist_idx = list(range(t0 - span, t0 + 1, stride))
        cand = np.where(np.all([usable(t) for t in hist_idx], axis=0) & at_init)[0]
        best_n = max(best_n, len(cand))
        if len(cand) >= num_points:
            break
    else:
        raise SkipVideo(f"at most {best_n} points visible over any history window and at frame "
                        f"{init_frame} (t0 in [{first}, {last}], need {num_points})")
    if init_frame > hist_idx[0]:
        raise SkipVideo(f"init frame {init_frame} is after the history start {hist_idx[0]}")
    chosen = cand[[int(i * len(cand) / num_points) for i in range(num_points)]]
    fut_idx = list(range(t0 + stride, min(T - 1, t0 + future * stride) + 1, stride))

    def to_cam(x: np.ndarray, t: int) -> np.ndarray:        # world -> camera frame at t
        w2c = np.linalg.inv(poses[t])
        return (x @ w2c[:3, :3].T + w2c[:3, 3]).astype(np.float32)

    points_3d_history = to_cam(pts_3d[chosen][:, hist_idx], t0).transpose(1, 0, 2)   # (H, P, 3)
    fut_raw = pts_3d[chosen][:, fut_idx]
    gt_future_3d = to_cam(np.nan_to_num(fut_raw), t0)                                # (P, F, 3)
    gt_future_vis = vis_3d[chosen][:, fut_idx] & np.isfinite(fut_raw).all(-1)

    episode = meta_in.get("episode", {})
    goal = meta_in["action"]
    step = step_at(episode, t0)
    caption = {"goal": goal, "step": step["text"] if step else None}[caption_src]
    if not caption:
        raise SkipVideo(f"no {caption_src} caption at t0={t0}")

    name = f"sharerobot_{vid}" + (f"_{tag}" if tag else "")
    out_dir = output_root / f"{name}{stride_suffix(stride)}"
    meta = save_example(
        out_dir, clip_frames=frames[hist_idx + fut_idx], fps=fps / stride,
        points_2d_at_t0=tracks_2d[t0, chosen], points_3d_history=points_3d_history,
        K=camera_K(intr, t0, img_wh), gt_future_3d=gt_future_3d, gt_future_vis=gt_future_vis,
        caption=caption,
        meta={
            "dataset": "sharerobot",
            "video": vid,
            "episode": episode.get("episode"),
            "source_dataset": episode.get("dataset"),
            "goal": goal,
            "sub_step_at_t0": step,
            "caption_source": caption_src,
            "t0_absolute": t0,
            "history_frame_indices": hist_idx,
            "future_frame_indices": fut_idx,
            "init_frame_index": init_frame,
            "point_indices": chosen.tolist(),
            "clip_range": [clip_s, clip_e],
            "offset": offset,
            "stride": stride,
            "native_fps": fps,
            "effective_fps": fps / stride,
            "data_root": str(data_root.resolve()),
            "camera_pose_constant": bool(np.allclose(poses[:T], poses[0], atol=1e-6)),
        },
    )

    # The init frame: model input for running from the start of the episode (H=1).
    Image.fromarray(frames[init_frame]).save(out_dir / "frame_init.jpg", quality=95)
    torch.save(torch.from_numpy(np.ascontiguousarray(tracks_2d[init_frame, chosen], dtype=np.float32)),
               out_dir / "points_2d_at_init.pt")
    torch.save(torch.from_numpy(to_cam(pts_3d[chosen][:, [init_frame]], init_frame).transpose(1, 0, 2)),
               out_dir / "points_3d_at_init.pt")
    torch.save(torch.from_numpy(camera_K(intr, init_frame, img_wh)), out_dir / "intrinsics_K_init.pt")

    print(f"[ok] {out_dir}  clip=[{clip_s},{clip_e}] t0={t0} init={init_frame} F={len(fut_idx)} "
          f"fps={fps / stride:g} caption=\"{caption}\" "
          f"history_motion={meta['history_motion_m'] * 100:.1f}cm "
          f"future_motion={meta['future_motion_m'] * 100:.1f}cm")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Video id, i.e. a folder of the pipeline work dir (e.g. bridge_3715).")
    src.add_argument("--num_videos", type=int, help="Randomly sample this many videos of the work dir.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="data_generation/run_pipeline.py work dir (one folder per video).")
    ap.add_argument("--output_root", type=Path, default=REPO_ROOT / "result" / "molmo_motion_input" / "sharerobot")
    ap.add_argument("--tag", default=None, help="Suffix of the output dir, e.g. the depth backend.")
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--future", type=int, default=32)
    ap.add_argument("--num_points", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0,
                    help="Earliest t0 = motion-clip start + offset (frames); t0 advances further while "
                         "fewer than --num_points points are visible over the history.")
    ap.add_argument("--min_future", type=int, default=8, help="Minimum number of future frames (model steps).")
    ap.add_argument("--t0", type=int, default=None, help="Use exactly this t0 (ignores --offset).")
    ap.add_argument("--init_frame", type=int, default=0,
                    help="Frame saved as frame_init.jpg (+ its query points), default the first one.")
    ap.add_argument("--caption", choices=["goal", "step"], default="goal",
                    help="Episode goal, or the sub-step active at t0.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Temporal subsampling of the episode frames (1 = native rate).")
    args = ap.parse_args()
    if args.stride < 1:
        ap.error("--stride must be >= 1")

    kwargs = dict(history=args.history, future=args.future, num_points=args.num_points,
                  offset=args.offset, min_future=args.min_future, t0_fixed=args.t0,
                  stride=args.stride, init_frame=args.init_frame, caption_src=args.caption, tag=args.tag)
    videos = list_videos(args.data_root)

    if args.video is not None:
        if args.video not in videos:
            raise SystemExit(f"video {args.video!r} has no final tracks in {args.data_root}")
        try:
            convert(args.video, args.data_root, args.output_root, **kwargs)
        except SkipVideo as e:
            raise SystemExit(f"cannot convert {args.video}: {e}")
        return

    order = np.random.RandomState(args.seed).permutation(len(videos))
    n_done = 0
    for i in order:
        if n_done >= args.num_videos:
            break
        try:
            convert(videos[i], args.data_root, args.output_root, **kwargs)
            n_done += 1
        except SkipVideo as e:
            print(f"[skip] {videos[i]}: {e}")
    print(f"converted {n_done}/{args.num_videos} videos into {args.output_root}")


if __name__ == "__main__":
    main()
