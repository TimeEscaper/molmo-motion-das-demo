#!/usr/bin/env python3
"""
MolmoMotion-1M data-generation pipeline — end-to-end driver.

Turns raw RGB video + an action description into object-grounded 3D point
trajectories and motion-coherent clips, exactly as described in the paper
(§3.1, §4.1) and the appendix.

Stages (run any contiguous subset with --start_stage / --end_stage):
    1  Grounding        Qwen3 + (Molmo2-8B recaption) + MolmoPoint + SAM3 + K-means,
                        or manually clicked query points (--query_points manual)
    2  Depth + camera   ViPE monocular SLAM or Depth Anything 3 (depth_backend: vipe | da3):
                        per-frame metric depth, intrinsics, poses; static_camera: true keeps
                        the depth/intrinsics but fixes the camera
    3  2D tracking      AllTracker dense point tracks on the query points
    4  3D lift          back-project the 2D tracks with depth + camera to a metric world frame
    5  Filter + smooth  consensus-gated trust weighting + ray-only smoothing; with
                        keep_objects: first / moving, only the first moving object / all
                        moving objects are kept; the others are removed from all outputs
                        (query points, 2D/3D tracks) and the rest re-filtered
    6  Clip             segment each video into motion-coherent clips
    7  Visualize        viz/<vid>.mp4: 2D tracks over the video + 3D trajectories

Every stage caches its output and is skipped if already present, so the pipeline
is freely resumable.

Work dir: one folder per video, the same whatever the depth backend / camera mode; only
what later stages or users read is kept (backend and tracker scratch lives in .tmp/ and is
removed):
    config.yaml                       the effective config of the (last) run
    <vid>/meta.json                   video id, action, source video, query-point mode and,
                                      for --episodes, the episode.json (goal, sub-steps, fps)
    <vid>/video.mp4                   the video all stages run on (a copy of the source, or
                                      its 480p re-encode with encode_480p: true)
    <vid>/query_points/*_f<frame>.npz query_points (N, 3) [frame, x, y], dim (2,) [H, W] in
                                      source-video pixels; one file per object and frame
                                      (+ <vid>_molmo2_meta.json: grounding prompts / points)
    <vid>/depth.zip                   per-frame metric depth, zipped EXR ("<frame:05d>.exr", Z),
                                      at the camera's image size
    <vid>/camera/intrinsics.npz       data (T, 4) [fx, fy, cx, cy], inds (T,), K (T, 3, 3),
                                      image_size_wh (2,) of depth / intrinsics
    <vid>/camera/pose.npz             data (T, 4, 4) camera-to-world used for the lift
                                      (identity with static_camera), inds (T,)
    <vid>/tracks_2d.npz               tracks (T, N, 2), visibility (T, N), dim, all query points
    <vid>/tracks_3d.npz               points_3d (N, T, 3) world frame, visibility (N, T, 1)
    <vid>/final_tracks/<vid>_{3d,2d}.npz  filtered + smoothed tracks (kept points only)
    <vid>/final_tracks/<vid>_filter_meta.npz  per-point raw / smoothed 3D, trust, keep mask
    <vid>/clips.json                  motion clips per object
    <vid>/viz.mp4                     2D tracks over the video (top) + 3D trajectories (bottom)

Inputs (one of --tasks / --episodes)
    --tasks     JSON list of {"video_id", "video_path", "action"} objects (optionally
                "query_points_dir" for --query_points manual)
    --episodes  episode dirs, or dirs of episode dirs, as written by
                scripts/data/extract_sharerobot.py: <video_id>.mp4 + goal.txt (the action)
                + episode.json (+ query_points/ for --query_points manual)
    --config    YAML hyperparameters + corpus prompt settings (see configs/)
    --work_dir  all artifacts are written here

Query points (stage 1)
    --query_points auto     (default) ground the object and sample points with the models above
    --query_points manual   import clicked points instead (scripts/click_query_points.py):
                            <episode dir>/query_points/*_obj<k>_f<frame>.npz with
                            query_points (N, 3) [frame, x, y] and dim (2,) [H, W] of the image
                            clicked on; coordinates are rescaled to the video by its width
                            (an aspect-preserving resize, possibly with a caption band below)

This driver hardcodes no machine-specific paths. It uses the active Python
(`sys.executable`) and the installed `vipe` console script; point HF_HOME at your
HuggingFace cache if you do not want the default (~/.cache/huggingface).

Example
    python run_pipeline.py \\
        --tasks examples/tasks_example.json \\
        --config configs/human_manipulation.yaml \\
        --work_dir ./runs/example

    # ShareRobot episodes, fixed camera, clicked query points
    python run_pipeline.py \\
        --episodes ../result/sharerobot_raw \\
        --config configs/sharerobot_static.yaml \\
        --query_points manual --work_dir ./runs/sharerobot_static
"""
import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parent
THIRD_PARTY = REPO_ROOT / "third_party"
PY = sys.executable  # run children with the same interpreter / env


# ──────────────────────────────────────────────────────────────────────────────
# Config + paths
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    # grounding
    "kmeans_k": 100,
    "max_frames": 0,            # 0 = all frames
    "agent": "hand",            # "hand" | "robot gripper"
    "point_prompt_template": "point to {obj} gripped and picked up by the {agent}",
    "recaption": True,
    "save_masks": False,
    "save_debug": False,
    # video / tracking
    "fps": 15,
    "encode_480p": True,        # False = feed the source video as-is (native res + fps)
    "alltracker_max_side": 512,
    "max_frame_groups": 5,
    # depth + camera
    "depth_backend": "vipe",    # "vipe" | "da3" (see DEPTH_BACKENDS)
    "static_camera": False,     # True = keep depth + intrinsics, camera-to-world = identity
    # filter + smooth (paper / App. A.6 defaults)
    "smooth_steps": 100,
    "smooth_lr": 0.05,
    "alpha": 1000.0,
    "lambda_reg": 1e-4,
    "z_thresh": 1.0,
    "n_anchors": 16,
    "gating_power": 2,
    "min_cluster_size": 10,
    "depth_step": 1,
    # object selection after stage 5
    "keep_objects": "all",       # "all" | "moving" (all objects that move) | "first" (the
                                 # first moving object in grounding order)
    "moving_ratio": 0.5,         # moving = motion >= ratio * motion of the most-moving object
    # clipping (paper §4.1)
    "clip_threshold": 0.02,
    "clip_min_gap": 5,
    "clip_min_clip_sec": 0.5,
    "clip_min_frames": None,
    # visualization (stage 7)
    "visualize": True,           # viz/<vid>.mp4: 2D tracks over the video + 3D trajectories
    # tooling
    "vipe_cmd": "vipe",
    "vipe_pipeline": "default",  # ViPE config (third_party/vipe/configs/pipeline/*.yaml)
    "da3_model": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",  # metric; CC BY-NC 4.0 weights
    "da3_process_res": 504,      # DA3 processing resolution (long side)
    "da3_max_depth": 20.0,       # clip depth (m); DA3 puts the sky at ~200 m
    "corpus": "molmomotion",
}


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        if yaml is None:
            raise RuntimeError("pyyaml is required to read --config; pip install pyyaml")
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg.update(user)
    return cfg


class Paths:
    """All artifact locations: run-level files in work_dir, everything about a video in its
    own folder work_dir/<vid>/ (layout in the module docstring)."""
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.tmp = work_dir / ".tmp"                     # scratch, removed after use
        self.tmp.mkdir(parents=True, exist_ok=True)

    def dir(self, vid):
        d = self.work_dir / vid
        (d / "camera").mkdir(parents=True, exist_ok=True)
        return d

    def video(self, vid, suffix=".mp4"):
        hits = sorted(self.dir(vid).glob("video.*"))
        return hits[0] if hits else self.dir(vid) / f"video{suffix}"

    def meta(self, vid):
        return self.dir(vid) / "meta.json"

    def qp_dir(self, vid):
        return self.dir(vid) / "query_points"

    def qp_files(self, vid):
        return sorted(self.qp_dir(vid).glob(f"{vid}_*_f*.npz"))

    def depth_zip(self, vid):
        return self.dir(vid) / "depth.zip"

    def intrinsics(self, vid):
        return self.dir(vid) / "camera" / "intrinsics.npz"

    def pose(self, vid):
        return self.dir(vid) / "camera" / "pose.npz"

    def tracks_2d(self, vid):
        return self.dir(vid) / "tracks_2d.npz"

    def tracks_3d(self, vid):
        return self.dir(vid) / "tracks_3d.npz"

    def final_dir(self, vid):                            # <vid>_{3d,2d,filter_meta}.npz, the
        d = self.dir(vid) / "final_tracks"               # names stages 5/6 scripts use
        d.mkdir(exist_ok=True)
        return d

    def final_3d(self, vid):
        return self.final_dir(vid) / f"{vid}_3d.npz"

    def final_2d(self, vid):
        return self.final_dir(vid) / f"{vid}_2d.npz"

    def filter_meta(self, vid):
        return self.final_dir(vid) / f"{vid}_filter_meta.npz"

    def clips(self, vid):
        return self.dir(vid) / "clips.json"

    def viz(self, vid):
        return self.dir(vid) / "viz.mp4"


def sh(cmd, **kw):
    """Run a command list, inheriting env. Returns the CompletedProcess."""
    return subprocess.run(cmd, env=os.environ.copy(), **kw)


def tool(cmd):
    """A console script: from PATH, else next to this interpreter (e.g. .venv/bin/vipe
    when the venv is used without being activated)."""
    if shutil.which(cmd):
        return cmd
    local = Path(PY).parent / cmd
    return str(local) if local.exists() else cmd


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def video_size(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    wh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return wh


def input_video(task, cfg, paths):
    """<vid>/video.mp4, the video ViPE + AllTracker run on: the 480p re-encode (written by
    Stage 2), or a copy of the source (the work dir is self-contained)."""
    vid = task["video_id"]
    if cfg["encode_480p"]:
        return paths.dir(vid) / "video.mp4"
    src = Path(task["video_path"]).resolve()
    dst = paths.dir(vid) / f"video{src.suffix}"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
    return dst


def write_run_info(tasks, cfg, args, paths):
    """config.yaml (the effective config) and <vid>/meta.json per video: the action, the
    source, the query-point mode and, for --episodes, the whole episode.json."""
    info = {"config_file": str(Path(args.config).resolve()) if args.config else None, **cfg}
    (paths.work_dir / "config.yaml").write_text(
        yaml.safe_dump(info, sort_keys=False) if yaml else json.dumps(info, indent=2))
    for task in tasks:
        vid = task["video_id"]
        if not cfg["encode_480p"]:
            input_video(task, cfg, paths)                 # copy the source into <vid>/
        meta = {
            "video_id": vid,
            "action": task["action"],
            "video": input_video(task, cfg, paths).name,
            "source_video": task["video_path"],
            "query_points": args.query_points,
            **({"episode_dir": task["episode_dir"], "episode": task["episode"]} if "episode" in task else {}),
        }
        paths.meta(vid).write_text(json.dumps(meta, indent=2) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — grounding
# ──────────────────────────────────────────────────────────────────────────────

def stage1_grounding(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 1: Grounding + query points\n", "=" * 80, sep="")
    todo = [t for t in tasks if not paths.qp_files(t["video_id"])]
    if not todo:
        print("  All tasks grounded.")
        return
    scratch = paths.tmp / "grounding"                 # the worker's <vid>/query_points/ layout
    tasks_json = paths.tmp / "grounding_tasks.json"
    prompts_json = paths.tmp / "grounding_prompts.json"
    config_json = paths.tmp / "grounding_config.json"
    tasks_json.write_text(json.dumps(todo))
    config_json.write_text(json.dumps(cfg))

    worker = REPO_ROOT / "pipeline" / "grounding_worker.py"
    # SAM 3 can rarely SIGSEGV; the worker is resumable, so retry a few times.
    for attempt in range(5):
        remaining = [t for t in todo
                     if not list((scratch / t["video_id"] / "query_points")
                                 .glob(f"{t['video_id']}_*_f*.npz"))]
        if not remaining:
            break
        print(f"  Attempt {attempt + 1}: {len(remaining)} task(s) remaining...")
        ret = sh([PY, str(worker), str(tasks_json), str(prompts_json),
                  str(scratch), str(config_json)])
        if ret.returncode == 0:
            break
        print(f"  worker exited {ret.returncode}; restarting (resumes from cache)")
    for t in todo:
        src = scratch / t["video_id"] / "query_points"
        if list(src.glob("*.npz")):
            dst = paths.qp_dir(t["video_id"])
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                shutil.move(str(f), dst / f.name)
    for f in (tasks_json, prompts_json, config_json):
        f.unlink(missing_ok=True)
    shutil.rmtree(scratch, ignore_errors=True)
    free_gpu()


def stage1_manual(tasks, cfg, paths):
    """Stage 1 replacement: copy clicked query points into query_points/<vid>/
    <vid>_manual_obj<k>_f<frame>.npz, in source-video pixels."""
    print("=" * 80, "\nSTAGE 1: Manual query points\n", "=" * 80, sep="")
    for i, task in enumerate(tasks):
        vid = task["video_id"]
        if paths.qp_files(vid):
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — query points exist ({paths.qp_dir(vid)})")
            continue
        src = task.get("query_points_dir")
        files = sorted(Path(src).glob("*_obj*_f*.npz")) if src else []
        if not files:
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — no *_obj<k>_f<frame>.npz in {src}")
            continue
        W, H = video_size(task["video_path"])
        dst = paths.qp_dir(vid)
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            obj, frame = re.search(r"_obj(\d+)_f(\d+)\.npz$", f.name).groups()
            qp = np.load(f)
            pts = qp["query_points"].astype(np.float64)
            s = W / float(qp["dim"][1])               # clicked image -> video, aspect-preserving
            pts[:, 1:] *= s
            inside = (pts[:, 1] >= 0) & (pts[:, 1] < W) & (pts[:, 2] >= 0) & (pts[:, 2] < H)
            if not inside.all():
                print(f"  WARN {f.name}: dropped {(~inside).sum()} point(s) outside the {W}x{H} video")
            np.savez(dst / f"{vid}_manual_obj{obj}_f{frame}.npz",
                     query_points=np.round(pts[inside]).astype(np.int32),
                     dim=np.array([H, W], dtype=np.int32))
            print(f"[{i+1}/{len(tasks)}] {vid}: obj{obj} f{frame}, {inside.sum()} points from {f}"
                  + (f" (x{s:.3f})" if s != 1 else ""))


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — depth + camera
# ──────────────────────────────────────────────────────────────────────────────

def _ffmpeg_encode_one(video_id, video_path, out, fps):
    import imageio_ffmpeg
    if out.exists():
        return video_id, "cached"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run(
        [ffmpeg, "-y", "-i", video_path, "-vf", "scale=-2:480", "-r", str(fps),
         "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an", str(out)],
        capture_output=True)
    if r.returncode != 0 or not out.exists():
        if out.exists():
            out.unlink()
        raise RuntimeError(f"ffmpeg failed for {video_id}: {r.stderr.decode(errors='replace')[-400:]}")
    return video_id, "encoded"


def depth_vipe(vid, video, cfg, scratch):
    """ViPE on `video` into `scratch`. Returns (depth zip, intrinsics (T, 4), camera-to-world
    (T, 4, 4), inds (T,), image size (w, h) of depth / intrinsics), or None on failure."""
    r = sh([tool(cfg["vipe_cmd"]), "infer", str(video), "--output", str(scratch),
            "--pipeline", cfg["vipe_pipeline"]])
    name = video.stem                                    # ViPE names artifacts after the file
    depth = scratch / "depth" / f"{name}.zip"
    if r.returncode != 0 or not depth.exists():
        return None
    intr, pose = np.load(scratch / "intrinsics" / f"{name}.npz"), np.load(scratch / "pose" / f"{name}.npz")
    return depth, intr["data"], pose["data"], intr["inds"], video_size(scratch / "rgb" / f"{name}.mp4")


_DA3 = {}   # loaded Depth Anything 3 model, kept for all videos of a Stage 2 run


def write_depth_zip(path, depths):
    """ViPE's depth artifact format: <frame:05d>.exr (half-float Z channel) in a zip."""
    import tempfile
    import zipfile

    import Imath
    import OpenEXR

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, d in enumerate(depths):
            header = OpenEXR.Header(d.shape[1], d.shape[0])
            header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
            with tempfile.NamedTemporaryFile(suffix=".exr") as f:
                exr = OpenEXR.OutputFile(f.name, header)
                exr.writePixels({"Z": d.astype(np.float16).tobytes()})
                exr.close()
                z.write(f.name, f"{i:05d}.exr")


def depth_da3(vid, video, cfg, scratch):
    """Depth Anything 3 on all frames of `video` in one multi-view pass: depth, intrinsics and
    poses jointly (DA3NESTED-* models are metric). Same return as depth_vipe; depth and
    intrinsics at the video resolution."""
    import cv2
    import torch
    from depth_anything_3.api import DepthAnything3

    cap, frames = cv2.VideoCapture(str(video)), []
    while (ok_frame := cap.read())[0]:
        frames.append(cv2.cvtColor(ok_frame[1], cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    H, W = frames[0].shape[:2]
    if cfg["da3_model"] not in _DA3:
        _DA3.clear()
        _DA3[cfg["da3_model"]] = DepthAnything3.from_pretrained(cfg["da3_model"]).to("cuda").eval()
    with torch.inference_mode():
        pred = _DA3[cfg["da3_model"]].inference(frames, process_res=cfg["da3_process_res"],
                                                process_res_method="upper_bound_resize")
    depth = np.asarray(pred.depth, dtype=np.float32)                 # (T, h, w) at processing res
    K = np.asarray(pred.intrinsics, dtype=np.float64)                # (T, 3, 3) at processing res
    T, h, w = depth.shape
    # "*resize" processing rescales the full frame (no crop): map back to the video resolution.
    sx, sy = W / w, H / h
    depth = np.clip([cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR) for d in depth],
                    0, cfg["da3_max_depth"])
    intr = np.stack([K[:, 0, 0] * sx, K[:, 1, 1] * sy, K[:, 0, 2] * sx, K[:, 1, 2] * sy], 1)
    w2c = np.tile(np.eye(4), (T, 1, 1))
    w2c[:, :3, :] = np.asarray(pred.extrinsics, dtype=np.float64)[:, :3, :4]   # OpenCV world-to-camera
    scratch.mkdir(parents=True, exist_ok=True)
    write_depth_zip(scratch / "depth.zip", depth)
    return scratch / "depth.zip", intr, np.linalg.inv(w2c), np.arange(T), (W, H)


# depth_backend -> fn(vid, video, cfg, scratch dir) -> (depth zip, intrinsics (T, 4),
# camera-to-world (T, 4, 4), inds (T,), image size (w, h)) or None; Stage 2 writes them
# to <vid>/depth.zip + <vid>/camera/, re-anchored to frame 0 (identity with static_camera).
DEPTH_BACKENDS = {"vipe": depth_vipe, "da3": depth_da3}


def stage2_depth_camera(tasks, cfg, paths):
    backend = cfg["depth_backend"]
    if backend not in DEPTH_BACKENDS:
        raise SystemExit(f"depth_backend {backend!r} not in {sorted(DEPTH_BACKENDS)}")
    label = backend + (f" ({cfg['vipe_pipeline']} pipeline)" if backend == "vipe" else "")
    print("=" * 80, f"\nSTAGE 2: depth + camera: {label}, "
          f"{'fixed' if cfg['static_camera'] else 'estimated'} camera\n", "=" * 80, sep="")
    if cfg["encode_480p"]:
        todo = [t for t in tasks if not (paths.dir(t["video_id"]) / "video.mp4").exists()]
        if todo:
            print(f"  Pre-encoding {len(todo)} video(s) to 480p...")
            with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
                futs = {pool.submit(_ffmpeg_encode_one, t["video_id"], t["video_path"],
                                    paths.dir(t["video_id"]) / "video.mp4", cfg["fps"]): t["video_id"]
                        for t in todo}
                for fut in as_completed(futs):
                    try:
                        vid, msg = fut.result()
                        print(f"    {vid}: {msg}")
                    except RuntimeError as e:
                        print(f"    {futs[fut]}: WARN {e}")

    for i, task in enumerate(tasks):
        vid = task["video_id"]
        if all(p.exists() for p in (paths.depth_zip(vid), paths.intrinsics(vid), paths.pose(vid))):
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — depth + camera exist")
            continue
        video = input_video(task, cfg, paths)
        if not video.exists():
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — input video missing ({video})")
            continue
        print(f"[{i+1}/{len(tasks)}] {backend} {vid}")
        scratch = paths.tmp / backend / vid
        out = DEPTH_BACKENDS[backend](vid, video, cfg, scratch)
        if out is None:
            print(f"  ERROR: {backend} failed")
            continue
        depth, intr, c2w, inds, wh = out
        c2w = np.linalg.inv(c2w[0])[None] @ c2w                 # world = camera at frame 0
        drift = float(np.linalg.norm(c2w[:, :3, 3], axis=1).max())
        if cfg["static_camera"]:
            c2w = np.tile(np.eye(4), (len(c2w), 1, 1))
        intr = intr.astype(np.float32)
        K = np.zeros((len(intr), 3, 3), np.float32)
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = *intr[:, :4].T, 1.0
        shutil.move(str(depth), paths.depth_zip(vid))
        np.savez(paths.intrinsics(vid), data=intr, inds=inds, K=K, image_size_wh=np.array(wh, np.int32))
        np.savez(paths.pose(vid), data=c2w.astype(np.float32), inds=inds)
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"  fx {intr[:, 0].min():.1f}-{intr[:, 0].max():.1f} px, {wh[0]}x{wh[1]}, "
              f"{backend} camera drift {100 * drift:.1f} cm"
              + (" (replaced by a fixed camera)" if cfg["static_camera"] else ""))
    shutil.rmtree(paths.tmp / backend, ignore_errors=True)
    _DA3.clear()
    free_gpu()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — AllTracker 2D tracking
# ──────────────────────────────────────────────────────────────────────────────

def _scale_qps(frame_qps, orig_dim, target_h, target_w):
    H, W = int(orig_dim[0]), int(orig_dim[1])
    if H == target_h and W == target_w:
        return frame_qps, orig_dim
    sx, sy = target_w / W, target_h / H
    scaled = frame_qps.copy()
    scaled[:, 1] = np.round(frame_qps[:, 1] * sx).astype(np.int32)
    scaled[:, 2] = np.round(frame_qps[:, 2] * sy).astype(np.int32)
    return scaled, np.array([target_h, target_w], dtype=np.int32)


def stage3_alltracker(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 3: AllTracker 2D tracking\n", "=" * 80, sep="")
    alltracker_dir = THIRD_PARTY / "alltracker"
    scratch_root = paths.tmp / "alltracker"           # <vid>/<clip>.npz, qp_frame_<f>.npz

    for i, task in enumerate(tasks):
        vid = task["video_id"]
        merged_file = paths.tracks_2d(vid)
        if merged_file.exists():
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — 2D tracks exist")
            continue
        video = input_video(task, cfg, paths)
        if not video.exists():
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — tracking video missing; run Stage 2")
            continue

        qp_files = paths.qp_files(vid)
        if not qp_files:
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — no query points")
            continue

        frame_groups, orig_dim = {}, None
        for qp_file in qp_files:
            m = re.search(r"_f(\d+)\.npz$", qp_file.name)
            if not m:
                continue
            try:
                qp = np.load(qp_file, allow_pickle=True)
            except (EOFError, ValueError) as e:
                print(f"  WARN corrupted {qp_file.name}: {e}")
                continue
            if orig_dim is None and "dim" in qp.files:
                orig_dim = qp["dim"]
            frame_groups.setdefault(int(m.group(1)), []).append(qp["query_points"])
        if orig_dim is None or not frame_groups:
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — query points empty/corrupt")
            continue

        H, W = int(orig_dim[0]), int(orig_dim[1])
        H480 = 480 if cfg["encode_480p"] else H
        W480 = (int(round(W * H480 / H / 2)) * 2) if cfg["encode_480p"] else W

        if len(frame_groups) > cfg["max_frame_groups"]:
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — {len(frame_groups)} frame groups "
                  f"> limit {cfg['max_frame_groups']}")
            continue

        scratch = scratch_root / vid
        scratch.mkdir(parents=True, exist_ok=True)
        frame_track_files = []
        for fidx in sorted(frame_groups):
            qps_orig = np.concatenate(frame_groups[fidx], axis=0)
            clip_name = f"{vid}_f{fidx}"
            track_file = scratch / f"{clip_name}.npz"
            if track_file.exists():
                frame_track_files.append((fidx, track_file))
                continue
            qps_480, dim_480 = _scale_qps(qps_orig, orig_dim, H480, W480)
            qp_480_file = scratch / f"qp_frame_{fidx}.npz"
            np.savez(qp_480_file, query_points=qps_480, dim=dim_480)
            print(f"[{i+1}/{len(tasks)}] AllTracker {vid} f{fidx}: {len(qps_orig)} pts "
                  f"({W}x{H} -> {W480}x{H480})")
            sh([PY, "run-query-points.py",
                "--file", vid, "--clip", clip_name,
                "--video_path", str(video), "--query_path", str(qp_480_file.resolve()),
                "--out_root", str(scratch_root.resolve()),
                "--max_frames", "0", "--max_side", str(cfg["alltracker_max_side"])],
               cwd=str(alltracker_dir))
            if track_file.exists():
                frame_track_files.append((fidx, track_file))
            else:
                print(f"  ERROR: AllTracker failed for f{fidx}")

        if frame_track_files and len(frame_track_files) == len(frame_groups):
            all_t, all_v, dim = [], [], None
            for _, tp in sorted(frame_track_files):
                td = np.load(tp, allow_pickle=True)
                all_t.append(td["tracks"])
                all_v.append(td["visibility"])
                dim = dim if dim is not None else td["dim"]
            np.savez(merged_file,
                     tracks=np.concatenate(all_t, axis=1),
                     visibility=np.concatenate(all_v, axis=1), dim=dim)
            shutil.rmtree(scratch, ignore_errors=True)
            print(f"  merged -> {merged_file}")
    shutil.rmtree(scratch_root, ignore_errors=True)
    free_gpu()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — 3D lift
# ──────────────────────────────────────────────────────────────────────────────

def _lift_one(vid, cfg, paths):
    """Back-project every visible 2D track point with the depth at its (rounded) pixel and
    the frame's intrinsics, then map it to the world frame with the camera-to-world pose."""
    from vipe.utils.io import read_depth_artifacts

    track_file, out_3d = paths.tracks_2d(vid), paths.tracks_3d(vid)
    if not track_file.exists():
        return vid, False, "no 2D tracks"
    if not paths.depth_zip(vid).exists():
        return vid, False, "no depth + camera"
    if out_3d.exists():
        return vid, True, "cached"
    d2 = np.load(track_file)
    tracks, vis, (h, w) = d2["tracks"], d2["visibility"] > 0, d2["dim"]   # (T, N, 2) in dim pixels
    intr, c2w = np.load(paths.intrinsics(vid))["data"], np.load(paths.pose(vid))["data"]
    T, N = tracks.shape[:2]
    points = np.full((N, T, 3), np.nan, dtype=np.float32)
    visibility = np.zeros((N, T, 1), dtype=bool)
    for t, (_, depth) in enumerate(read_depth_artifacts(paths.depth_zip(vid))):
        if t >= T:
            break
        if t % cfg["depth_step"] or not vis[t].any():
            continue
        depth = depth.numpy()
        dh, dw = depth.shape
        xy = tracks[t, vis[t]] * np.array([dw / w, dh / h])       # tracks -> depth pixels
        x = np.clip(np.round(xy[:, 0]).astype(int), 0, dw - 1)
        y = np.clip(np.round(xy[:, 1]).astype(int), 0, dh - 1)
        z = depth[y, x]
        ok = np.isfinite(z) & (z > 0)
        fx, fy, cx, cy = intr[t, :4]
        cam = np.stack([(x[ok] - cx) * z[ok] / fx, (y[ok] - cy) * z[ok] / fy, z[ok]], axis=1)
        idx = np.where(vis[t])[0][ok]
        points[idx, t] = cam @ c2w[t, :3, :3].T + c2w[t, :3, 3]
        visibility[idx, t, 0] = True
    np.savez_compressed(out_3d, points_3d=points, visibility=visibility)
    return vid, True, f"done ({visibility.any(axis=(1, 2)).sum()}/{N} points lifted)"


def stage4_lift(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 4: 3D lift\n", "=" * 80, sep="")
    with ThreadPoolExecutor(max_workers=min(16, len(tasks))) as pool:
        futs = {pool.submit(_lift_one, t["video_id"], cfg, paths): t["video_id"] for t in tasks}
        for fut in as_completed(futs):
            vid, ok, msg = fut.result()
            print(f"  [{'OK' if ok else 'WARN'}] {vid}: {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — filter + smooth
# ──────────────────────────────────────────────────────────────────────────────

def _run_filter(vid, cfg, paths):
    out_3d = paths.tracks_3d(vid)
    final_3d = paths.final_3d(vid)
    # object_sizes = number of query points per grounded object (for sub-grouping)
    sizes = [len(np.load(f, allow_pickle=True)["query_points"]) for f in paths.qp_files(vid)]
    script = THIRD_PARTY / "vipe" / "track-filter-smooth.py"
    cmd = [PY, str(script), "--vid", vid,
           "--src", str(out_3d), "--dst", str(final_3d),
           "--tracks_2d", str(paths.tracks_2d(vid)),
           "--tracks_2d_out", str(paths.final_2d(vid)),
           "--pose_npz", str(paths.pose(vid)),
           "--depth_zip", str(paths.depth_zip(vid)), "--intrinsics_npz", str(paths.intrinsics(vid)),
           "--smooth_steps", str(cfg["smooth_steps"]), "--smooth_lr", str(cfg["smooth_lr"]),
           "--alpha", str(cfg["alpha"]), "--lambda_reg", str(cfg["lambda_reg"]),
           "--z_thresh", str(cfg["z_thresh"]), "--n_anchors", str(cfg["n_anchors"]),
           "--gating_power", str(cfg["gating_power"])]
    # NOTE: mean-shift min_cluster_size is fixed to the paper default (10) inside
    # track-filter-smooth.py; it is not a CLI flag.
    if sizes:
        cmd += ["--object_sizes", ",".join(map(str, sizes))]
    r = sh(cmd)
    return r.returncode == 0 and final_3d.exists(), r.returncode


def _filter_smooth_one(vid, cfg, paths):
    final_3d, final_2d = paths.final_3d(vid), paths.final_2d(vid)
    if not paths.tracks_3d(vid).exists():
        return vid, False, "no 3D tracks"
    if final_3d.exists() and final_2d.exists():
        return vid, True, "cached"
    ok, code = _run_filter(vid, cfg, paths)
    if not ok:
        return vid, False, f"failed ({code})"
    if cfg["keep_objects"] not in ("moving", "first"):
        return vid, True, "done"
    # Keep only the manipulated object(s): drop the others from every output and re-filter.
    motion, moving = moving_objects(vid, cfg, paths)
    order = grounding_order(vid, list(motion), paths)
    if cfg["keep_objects"] == "first":
        first = next((o for o in order if o in moving), None)
        keep = {first} if first else moving
    else:
        keep = moving
    reasons = {o: "not moving" if o not in moving else "not the first moving object"
               for o in motion if o not in keep}
    msg = "objects " + ", ".join(f"{o} {100 * motion[o]:.1f} cm" + (f" (dropped: {reasons[o]})" if o in reasons else "")
                                 for o in order)
    if reasons:
        drop_objects(vid, reasons, motion, paths)
        for f in paths.final_dir(vid).glob(f"{vid}_*.npz"):
            f.unlink()
        ok, code = _run_filter(vid, cfg, paths)
        if not ok:
            return vid, False, f"{msg}; re-filter failed ({code})"
    n = len(np.load(final_3d, allow_pickle=True)["points_3d"])
    return vid, True, f"done; {msg} -> {n} tracks"


def track_objects(vid, paths):
    """Query-point file (= one grounded object at one detection frame) of every 2D/3D track,
    in Stage 3's order: frame groups by frame index, files sorted within a group."""
    groups = {}
    for f in paths.qp_files(vid):
        groups.setdefault(int(re.search(r"_f(\d+)\.npz$", f.name).group(1)), []).append(f)
    names, frames = [], []
    for fidx in sorted(groups):
        for f in groups[fidx]:
            n = len(np.load(f, allow_pickle=True)["query_points"])
            names += [f.stem[len(vid) + 1:]] * n
            frames += [fidx] * n
    return np.array(names), np.array(frames)


def moving_objects(vid, cfg, paths):
    """Motion of every grounded object -- the median over its kept, smoothed tracks of the
    largest 3D displacement from the query frame -- and the objects that move, i.e. at least
    `moving_ratio` x the largest motion (the manipulated ones; grounding can also name e.g.
    the destination in "move the croissant to the towel")."""
    meta = np.load(paths.filter_meta(vid), allow_pickle=True)
    names, frames = track_objects(vid, paths)
    P, keep = meta["P_smoothed"], meta["keep_mask"].astype(bool)
    if P.dtype == object or len(names) != len(P):
        names = np.array([o for o in dict.fromkeys(names)] or ["all"])
        return {o: 0.0 for o in names}, set(names)               # cannot attribute: keep all
    ref = P[np.arange(len(P)), np.minimum(frames, P.shape[1] - 1)]          # position at query frame
    with np.errstate(invalid="ignore"):
        disp = np.nanmax(np.linalg.norm(P - ref[:, None], axis=-1), axis=1)  # (N,)
    motion = {o: float(np.nanmedian(disp[keep & (names == o)])) if (keep & (names == o)).any() else 0.0
              for o in dict.fromkeys(names)}
    top = max(motion.values())
    return motion, {o for o, m in motion.items() if m >= cfg["moving_ratio"] * top}


def object_matches(name, key):
    """Does a grounding meta object name ("metal pot") belong to a query-point file key
    ("molmo2_metalpot_f0_f0"; grounding drops the spaces)?"""
    return re.sub(r"[^0-9a-z]", "", name.lower()) in key.lower().split("_")


def grounding_order(vid, keys, paths):
    """Query-point file keys in the order grounding named their objects (the per_object list
    of the grounding meta json; e.g. "croissant | towel"), then the rest sorted (for manual
    points: obj0, obj1, ...)."""
    ordered = []
    for meta_file in paths.qp_dir(vid).glob("*_meta.json"):
        for e in json.loads(meta_file.read_text()).get("per_object", []):
            ordered += [k for k in keys if object_matches(e["object"], k) and k not in ordered]
    return ordered + sorted(k for k in keys if k not in ordered)


def drop_objects(vid, reasons, motion, paths):
    """Remove objects ({key: reason}) from query_points/ (their files, and their entries in the
    grounding meta json, which lists them under "dropped_objects"), tracks_2d/ and tracks_3d/."""
    dropped = set(reasons)
    names, _ = track_objects(vid, paths)
    keep = ~np.isin(names, list(dropped))
    t2 = dict(np.load(paths.tracks_2d(vid)))
    t2["tracks"], t2["visibility"] = t2["tracks"][:, keep], t2["visibility"][:, keep]
    np.savez(paths.tracks_2d(vid), **t2)
    t3 = dict(np.load(paths.tracks_3d(vid)))
    t3["points_3d"], t3["visibility"] = t3["points_3d"][keep], t3["visibility"][keep]
    np.savez_compressed(paths.tracks_3d(vid), **t3)
    for o in dropped:
        (paths.qp_dir(vid) / f"{vid}_{o}.npz").unlink()
    for meta_file in paths.qp_dir(vid).glob("*_meta.json"):
        meta = json.loads(meta_file.read_text())
        kept, gone = [], []
        for e in meta.get("per_object", []):
            hit = [o for o in dropped if object_matches(e["object"], o)]
            if hit:
                gone.append({**e, "reason": reasons[hit[0]], "motion_m": round(motion[hit[0]], 4)})
            else:
                kept.append(e)
        meta["per_object"] = kept
        meta["dropped_objects"] = meta.get("dropped_objects", []) + gone
        meta_file.write_text(json.dumps(meta, indent=2))


def reprojection_error(vid, paths):
    """Median distance (px) between the final 3D tracks projected with camera/ and the final
    2D tracks (rescaled to the camera image size), or None if unavailable."""
    d3 = np.load(paths.final_3d(vid), allow_pickle=True)
    d2 = np.load(paths.final_2d(vid), allow_pickle=True)
    if d3["points_3d"].dtype == object:  # nested per-object format
        return None
    intr = np.load(paths.intrinsics(vid))
    K, wh, c2w = intr["K"], intr["image_size_wh"], np.load(paths.pose(vid))["data"]
    P = d3["points_3d"]                                       # (N, T, 3) world
    uv_gt, vis = d2["tracks"], d2["visibility"].astype(bool)  # (T, N, 2), (T, N)
    uv_gt = uv_gt * (wh / d2["dim"][::-1])                     # track pixels -> camera pixels
    T = min(P.shape[1], len(K), len(c2w), len(uv_gt))
    w2c = np.linalg.inv(c2w[:T].astype(np.float64))
    Pc = np.einsum("tij,ntj->nti", w2c[:, :3, :3], P[:, :T]) + w2c[None, :, :3, 3]
    uv = np.einsum("tij,ntj->nti", K[:T], Pc)
    uv = uv[..., :2] / uv[..., 2:3]
    err = np.linalg.norm(uv - uv_gt[:T].transpose(1, 0, 2), axis=-1)
    ok = vis[:T].T & np.isfinite(err)
    return float(np.median(err[ok])) if ok.any() else None


def stage5_filter_smooth(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 5: Filter + smooth\n", "=" * 80, sep="")
    with ThreadPoolExecutor(max_workers=min(16, len(tasks))) as pool:
        futs = {pool.submit(_filter_smooth_one, t["video_id"], cfg, paths): t["video_id"] for t in tasks}
        for fut in as_completed(futs):
            vid, ok, msg = fut.result()
            if ok:
                err = reprojection_error(vid, paths)
                msg += f"; reproj. err {err:.2f} px" if err is not None else ""
            print(f"  [{'OK' if ok else 'WARN'}] {vid}: {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 6 — clipping
# ──────────────────────────────────────────────────────────────────────────────

def stage6_clip(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 6: Video-level motion clipping\n", "=" * 80, sep="")
    vids = [t["video_id"] for t in tasks if paths.final_3d(t["video_id"]).exists()]
    if not vids:
        print("  No final tracks to clip.")
        return
    script = REPO_ROOT / "pipeline" / "clip_segments.py"
    for vid in vids:                                     # -> <vid>/clips.json
        cmd = [PY, str(script),
               "--final_tracks_dir", str(paths.final_dir(vid)),
               "--video_ids", vid,
               "--out_json", str(paths.clips(vid)),
               "--fps", str(cfg["fps"]),
               "--threshold", str(cfg["clip_threshold"]),
               "--min_gap", str(cfg["clip_min_gap"]),
               "--min_clip_sec", str(cfg["clip_min_clip_sec"])]
        if cfg["clip_min_frames"] is not None:
            cmd += ["--min_frames", str(cfg["clip_min_frames"])]
        sh(cmd)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 7 — visualization
# ──────────────────────────────────────────────────────────────────────────────

def stage7_visualize(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 7: Visualization\n", "=" * 80, sep="")
    if not cfg["visualize"]:
        print("  visualize: false")
        return
    for i, task in enumerate(tasks):
        vid = task["video_id"]
        final, out = paths.filter_meta(vid), paths.viz(vid)
        if not final.exists():
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — no final tracks")
            continue
        if out.exists() and out.stat().st_mtime >= final.stat().st_mtime:
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — {out} is up to date")
            continue
        sh([PY, str(REPO_ROOT / "scripts" / "visualize_tracks.py"),
            "--video_dir", str(paths.dir(vid)), "--out", str(out)])


# ──────────────────────────────────────────────────────────────────────────────

STAGES = {
    1: ("Grounding", stage1_grounding),
    2: ("Depth + camera", stage2_depth_camera),
    3: ("AllTracker 2D", stage3_alltracker),
    4: ("3D lift", stage4_lift),
    5: ("Filter + smooth", stage5_filter_smooth),
    6: ("Clip", stage6_clip),
    7: ("Visualize", stage7_visualize),
}


def episode_tasks(roots):
    """Tasks from episode dirs (<video_id>.mp4 + goal.txt + episode.json), each given
    directly or as a dir of episode dirs."""
    tasks = []
    for root in map(Path, roots):
        dirs = [root] if (root / "episode.json").exists() else sorted(
            d for d in root.iterdir() if (d / "episode.json").exists())
        if not dirs:
            raise SystemExit(f"--episodes {root}: no episode.json in it or its subdirs")
        for d in dirs:
            meta = json.loads((d / "episode.json").read_text())
            video = d / meta["video"] if "video" in meta else next(d.glob("*.mp4"))
            goal = d / "goal.txt"
            tasks.append({
                "video_id": meta.get("video_id", video.stem),
                "video_path": str(video.resolve()),
                "action": goal.read_text().strip() if goal.exists() else meta["goal"],
                "query_points_dir": str((d / "query_points").resolve()),
                "episode_dir": str(d.resolve()),
                "episode": meta,
            })
    return tasks


def main():
    sys.stdout.reconfigure(line_buffering=True)   # keep our lines in order with the children's
    ap = argparse.ArgumentParser(description="MolmoMotion-1M data-generation pipeline")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tasks", help="JSON list of {video_id, video_path, action}")
    src.add_argument("--episodes", nargs="+",
                     help="Episode dirs (<video_id>.mp4 + goal.txt + episode.json) or dirs of them.")
    ap.add_argument("--config", default=None, help="YAML config (see configs/)")
    ap.add_argument("--work_dir", required=True, help="Output directory for all artifacts")
    ap.add_argument("--query_points", choices=["auto", "manual"], default="auto",
                    help="Stage 1: ground + sample query points with the models (auto), or import "
                         "clicked ones from each episode's query_points/ dir (manual).")
    ap.add_argument("--start_stage", type=int, default=1)
    ap.add_argument("--end_stage", type=int, default=7)
    args = ap.parse_args()

    if args.tasks:
        tasks = json.load(open(args.tasks))
        for t in tasks:
            if "action" not in t and "language_instruction" in t:
                t["action"] = t["language_instruction"]
    else:
        tasks = episode_tasks(args.episodes)
    cfg = load_config(args.config)
    paths = Paths(Path(args.work_dir).resolve())
    if args.query_points == "manual":
        STAGES[1] = ("Manual query points", stage1_manual)
    write_run_info(tasks, cfg, args, paths)

    print(f"\nMolmoMotion-1M data-gen: {len(tasks)} task(s), query points: {args.query_points}, "
          f"stages {args.start_stage}-{args.end_stage}, work_dir={paths.work_dir}\n")

    t0 = time.time()
    for s in range(args.start_stage, args.end_stage + 1):
        name, fn = STAGES[s]
        ts = time.time()
        fn(tasks, cfg, paths)
        print(f"  >> Stage {s} ({name}) finished in {time.time() - ts:.1f}s\n", flush=True)
    if not any(paths.tmp.iterdir()):
        paths.tmp.rmdir()
    print(f"PIPELINE DONE in {time.time() - t0:.1f}s. Outputs -> {paths.work_dir}/<video_id>/\n")


if __name__ == "__main__":
    main()
