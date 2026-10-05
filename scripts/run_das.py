"""Diffusion as Shader from a MolmoMotion prediction: tracking video + generated video.

Diffusion as Shader (DaS, arXiv:2501.03847; code in the `das/` submodule) generates a
49-frame 720x480 video from a first frame, a prompt and a "3D tracking video": 3D points
of the first frame, coloured by where they start (R = image column, G = row, B = 1/depth),
projected into every frame. MolmoMotion predicts P query points x F future steps in metric
camera coordinates. Per example this script:

  1. Depth: Depth Anything 3 on the input frame, scaled so the depth at the query pixels
     matches the z of the query points (their 3D position at the input frame).
  2. Mask: SAM 3 (tracker, point prompts) on the query points, united with one mask per
     extra click from `mask_points.json` in the example (parts that move with the query
     points but are segmented separately, e.g. the bike under a rider) or `--mask-points`;
     restricted to pixels within `--mask-depth-tol` of the query points' depth.
  3. Motion: the object's pixels move with the predicted 3D displacement of the query points
     (inverse-distance weights); the background stays put (fixed camera). The F + 1 steps
     are spread over DaS's 49 frames.
  4. Tracking video like DaS's real-video training data: SpaTracker's 70x70 grid of points,
     each a ~7x5 rectangle with black gaps, nearer points on top. Static points behind the
     object (inpainted depth) fill the area it uncovers (`--no-fill-background` to drop).
  5. DaS (CogVideoX-5B I2V + tracking ControlNet, `weights/Diffusion-As-Shader`) on the
     input frame, the caption and the tracking video.

The intrinsics must match the query points (the 3D points must project onto the 2D ones);
the reprojection error is reported, e.g. examples/data/davis_bmx_trees_das has a corrected K.

Inputs: an example dir (as written by scripts/data/sample_*.py) and the prediction of
scripts/run_molmo_motion.py, `<prediction-root>/<example>/[<run-name>_]prediction.pt`.
With `--init-frame` the input frame is the example's init frame (frame_init.jpg, as for
`run_molmo_motion.py --init-frame`), else t0 (frame_t+0.jpg).

Run::

    python scripts/run_molmo_motion.py --input examples/data/davis_bmx_trees_das --history 3 --run-name h3
    python scripts/run_das.py --input examples/data/davis_bmx_trees_das --run-name h3

Output, in ``<output>/<example>/`` (default ``result/das/``), prefixed with the run name:
``tracking.mp4`` (the DaS tracking video), ``tracking_overlay.mp4`` (blended over the input
frame + the query tracks), ``mask_overlay.png``, ``result.mp4`` (the generated video, re-timed
to the prediction's rate), ``comparison.mp4`` (tracking overlay | result) and ``das.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "das"))                                       # DaS models
sys.path.insert(0, str(_REPO_ROOT / "data_generation" / "third_party" / "sam3"))   # SAM 3 (source)

DAS_H, DAS_W, DAS_FRAMES = 480, 720, 49        # fixed by the DaS checkpoint (CogVideoX-5B I2V)
DAS_NEGATIVE_PROMPT = ("The video is not of a high quality, it has a low resolution. Watermark present "
                       "in each frame. The background is solid. Strange body and strange trajectory. "
                       "Distortion.")


# ──────────────────────────────────────────────────────────────────────────
# Inputs.
# ──────────────────────────────────────────────────────────────────────────

def load_inputs(example_dir: Path, prediction: Path, init_frame: bool) -> dict:
    if not prediction.exists():
        raise SystemExit(f"{prediction} not found; run scripts/run_molmo_motion.py on {example_dir} first")
    meta = json.loads((example_dir / "meta.json").read_text())
    caption = example_dir / "caption.txt"
    if init_frame:
        image, pts2d = example_dir / "frame_init.jpg", "points_2d_at_init.pt"
        pts3d, K = "points_3d_at_init.pt", "intrinsics_K_init.pt"
    else:
        image, pts2d, pts3d, K = example_dir / "frame_t+0.jpg", "points_2d_at_t0.pt", "points_3d_history.pt", "intrinsics_K.pt"
    mask_points = example_dir / "mask_points.json"
    return {
        "image_path": image,
        "image": Image.open(image).convert("RGB"),
        "K": torch.load(example_dir / K).double().numpy(),
        "points_2d": torch.load(example_dir / pts2d).double().numpy(),              # (P, 2)
        "anchor_3d": torch.load(example_dir / pts3d)[-1].double().numpy(),          # (P, 3)
        "pred_3d": torch.load(prediction).double().numpy(),                         # (P, F, 3)
        "caption": caption.read_text().strip() if caption.exists() else meta["caption"],
        "mask_points": json.loads(mask_points.read_text())["points"] if mask_points.exists() else [],
    }


def project(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(..., 3) camera-frame points -> (..., 2) pixels."""
    z = np.clip(xyz[..., 2], 1e-6, None)
    return np.stack([K[0, 0] * xyz[..., 0] / z + K[0, 2], K[1, 1] * xyz[..., 1] / z + K[1, 2]], axis=-1)


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────
# Depth (Depth Anything 3) and mask (SAM 3).
# ──────────────────────────────────────────────────────────────────────────

def da3_depth(image: Image.Image, model_id: str, process_res: int) -> np.ndarray:
    """(H, W) metric depth of one image at its resolution."""
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id).to("cuda").eval()
    with torch.inference_mode():
        pred = model.inference([np.asarray(image)], process_res=process_res,
                               process_res_method="upper_bound_resize")
    del model
    free_gpu()
    depth = np.asarray(pred.depth, dtype=np.float32)[0]
    return cv2.resize(depth, image.size, interpolation=cv2.INTER_NEAREST).astype(np.float64)


def sam3_mask(image: Image.Image, points_2d: np.ndarray, extra_points: list) -> np.ndarray:
    """SAM 3 tracker mask of the object under the query points (positive clicks), united with
    one mask per extra click (parts SAM segments as separate objects)."""
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    W, H = image.size
    mask = np.zeros((H, W), dtype=bool)
    with tempfile.TemporaryDirectory() as frames:              # same calls as the grounding stage
        image.save(Path(frames) / "0.jpg", quality=95)            # a one-frame "video"
        state = tracker.init_state(video_path=frames, offload_video_to_cpu=True)
        for obj_id, pts in enumerate([points_2d.tolist()] + [[p] for p in extra_points], start=1):
            pts = torch.tensor(pts, dtype=torch.float32) / torch.tensor([W, H], dtype=torch.float32)
            _, obj_ids, _, masks = tracker.add_new_points(
                inference_state=state, frame_idx=0, obj_id=obj_id, points=pts,
                labels=torch.ones(len(pts), dtype=torch.int32), clear_old_points=True)
            mask |= (masks[list(obj_ids).index(obj_id)] > 0).cpu().numpy().squeeze()
    del model, tracker
    free_gpu()
    return mask


# ──────────────────────────────────────────────────────────────────────────
# Tracking video: SpaTracker grid (DaS's real-video training data).
# ──────────────────────────────────────────────────────────────────────────

def spatracker_grid(density: int) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of SpaTracker's `get_points_on_a_grid(density, (384, 576))`, scaled to 720x480."""
    interp_h, interp_w = 384, 576
    step = interp_w // 64
    gy, gx = np.meshgrid(np.arange(density), np.arange(density), indexing="ij")
    v = step + gy / (density - 1) * (interp_h - 2 * step)
    u = step + gx / (density - 1) * (interp_w - 2 * step)
    return u.reshape(-1) * DAS_W / interp_w, v.reshape(-1) * DAS_H / interp_h


def track_colors(u: np.ndarray, v: np.ndarray, z: np.ndarray, n_front: int) -> np.ndarray:
    """(N, 3) uint8 as SpaTracker's `Visualizer` "rainbow" mode: R = u, G = v (normalized over
    the grid), B = 1/z normalized to its 2-98 percentiles; rows from `n_front` on (the
    background layer) use the statistics of the first `n_front` points."""
    uf, vf, inv_z = u[:n_front], v[:n_front], 1 / z
    p2, p98 = np.percentile(inv_z[:n_front], 2), np.percentile(inv_z[:n_front], 98)
    colors = np.stack([(u - uf.min()) / (uf.max() - uf.min()), (v - vf.min()) / (vf.max() - vf.min()),
                       (inv_z - p2) / (p98 - p2)], axis=-1)
    return (np.clip(colors, 0, 1) * 255).astype(np.uint8)


def background_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(H, W) depth of the static background hidden behind the object: inpainted from around
    a dilated mask and kept behind the object, so the first frame renders unchanged."""
    hole = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
    filled = cv2.inpaint(depth.astype(np.float32), hole, 5, cv2.INPAINT_TELEA).astype(np.float64)
    return np.maximum(filled, depth * 1.02)


def render_rects(uv: np.ndarray, z: np.ndarray, colors: np.ndarray, half: float) -> np.ndarray:
    """SpaTracker `Visualizer`: a filled rectangle per point, `half` wide and `half / 1.5` high
    on each side, drawn far to near."""
    img = np.zeros((DAS_H, DAS_W, 3), dtype=np.uint8)
    keep = np.isfinite(z) & (z > 0) & np.isfinite(uv).all(axis=1)
    for i in np.where(keep)[0][np.argsort(-z[keep], kind="stable")]:
        u, v = uv[i]
        cv2.rectangle(img, (int(u - half), int(v - half / 1.5)), (int(u + half), int(v + half / 1.5)),
                      colors[i].tolist(), thickness=-1)
    return img


def draw_tracks(frame: np.ndarray, tracks: np.ndarray, upto: int) -> np.ndarray:
    """Query-point tracks (P, T, 2) up to frame `upto`, drawn over `frame`."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for track in tracks:
        pts = [tuple(p) for p in track[:upto + 1]]
        if len(pts) > 1:
            draw.line(pts, fill=(255, 255, 255), width=2)
        x, y = pts[-1]
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 40, 40), outline=(255, 255, 255))
    return np.asarray(img)


def write_video(path: Path, frames, fps: float) -> None:
    imageio.mimsave(path, list(frames), fps=fps, codec="libx264", macro_block_size=1, quality=9)


def build_tracking(inp: dict, depth: np.ndarray, mask: np.ndarray, args) -> tuple[np.ndarray, list, dict]:
    """(49, 480, 720, 3) uint8 tracking video, its overlay frames, and diagnostics."""
    image, K, anchor, pred = inp["image"], inp["K"], inp["anchor_3d"], inp["pred_3d"]
    W0, H0 = image.size
    K_das = np.diag([DAS_W / W0, DAS_H / H0, 1.0]) @ K          # DaS stretches the frame to 720x480
    depth_g = cv2.resize(depth, (DAS_W, DAS_H), interpolation=cv2.INTER_NEAREST)
    mask_g = cv2.resize(mask.astype(np.uint8), (DAS_W, DAS_H), interpolation=cv2.INTER_NEAREST) > 0

    u0, v0 = spatracker_grid(args.density)
    half = int(min(np.diff(np.unique(u0)).min(), np.diff(np.unique(v0)).min())) / 2
    row, col = np.clip(v0.astype(int), 0, DAS_H - 1), np.clip(u0.astype(int), 0, DAS_W - 1)
    z0, obj = depth_g[row, col], mask_g[row, col]
    xyz = np.stack([(u0 - K_das[0, 2]) / K_das[0, 0] * z0, (v0 - K_das[1, 2]) / K_das[1, 1] * z0, z0], -1)

    # Motion: predicted displacements of the query points, steps 0..F spread over 49 frames,
    # spread to the object's points by inverse squared distance.
    F = pred.shape[1]
    disp = np.concatenate([np.zeros_like(anchor[:, None]), pred - anchor[:, None]], axis=1)  # (P, F+1, 3)
    steps = np.arange(DAS_FRAMES) * F / (DAS_FRAMES - 1)
    lo = np.floor(steps).astype(int)
    w = (steps - lo)[None, :, None]
    disp = disp[:, lo] * (1 - w) + disp[:, np.minimum(lo + 1, F)] * w                      # (P, 49, 3)
    d2 = ((xyz[obj][:, None] - anchor[None]) ** 2).sum(-1)
    wts = 1.0 / (d2 + 1e-6)
    wts /= wts.sum(1, keepdims=True)
    moved = xyz[obj][None] + np.einsum("mp,ptc->tmc", wts, disp)                              # (49, M, 3)

    if args.fill_background:
        bg_z = background_depth(depth_g, mask_g)[row[obj], col[obj]]
        bg_xyz = xyz[obj] * (bg_z / z0[obj])[:, None]                                        # same rays, farther
    else:
        bg_z, bg_xyz = np.zeros(0), np.zeros((0, 3))
    colors = track_colors(np.r_[u0, u0[obj][:len(bg_z)]], np.r_[v0, v0[obj][:len(bg_z)]],
                          np.r_[z0, bg_z], len(u0))
    image_g = np.asarray(image.resize((DAS_W, DAS_H), Image.BILINEAR))
    query_tracks = project(anchor[:, None] + disp, K_das)                                    # (P, 49, 2)
    frames, overlays = [], []
    for k in range(DAS_FRAMES):
        pts = xyz.copy()
        pts[obj] = moved[k]
        pts = np.concatenate([pts, bg_xyz])
        frames.append(render_rects(project(pts, K_das), pts[:, 2], colors, half))
        overlays.append(draw_tracks((0.5 * image_g + 0.5 * frames[-1]).astype(np.uint8), query_tracks, k))
    shift = np.linalg.norm(query_tracks[:, -1] - query_tracks[:, 0], axis=-1).mean()
    return np.stack(frames), overlays, {"grid_points": len(u0), "moving_points": int(obj.sum()),
                                        "background_points": len(bg_z), "future_steps": F,
                                        "query_motion_px": float(shift)}


# ──────────────────────────────────────────────────────────────────────────
# Diffusion as Shader.
# ──────────────────────────────────────────────────────────────────────────

def das_generate(checkpoint: Path, image: Image.Image, prompt: str, tracking: np.ndarray, args) -> list:
    """DaS's `DiffusionAsShaderPipeline._infer` (das/models/pipelines.py) without its
    tracking / depth / repainting dependencies: 49 PIL frames at 720x480."""
    from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
    from transformers import T5EncoderModel, T5Tokenizer
    from models.cogvideox_tracking import CogVideoXImageToVideoPipelineTracking, CogVideoXTransformer3DModelTracking

    dtype, device = torch.bfloat16, "cuda"
    pipe = CogVideoXImageToVideoPipelineTracking(
        vae=AutoencoderKLCogVideoX.from_pretrained(checkpoint, subfolder="vae"),
        text_encoder=T5EncoderModel.from_pretrained(checkpoint, subfolder="text_encoder"),
        tokenizer=T5Tokenizer.from_pretrained(checkpoint, subfolder="tokenizer"),
        transformer=CogVideoXTransformer3DModelTracking.from_pretrained(checkpoint, subfolder="transformer"),
        scheduler=CogVideoXDDIMScheduler.from_pretrained(checkpoint, subfolder="scheduler"))
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device, dtype=dtype)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.transformer.gradient_checkpointing = False

    maps = torch.from_numpy(tracking).permute(0, 3, 1, 2).float().div(255).to(device, dtype)  # (T, C, H, W)
    with torch.inference_mode():
        latents = pipe.vae.encode(maps.unsqueeze(0).permute(0, 2, 1, 3, 4)).latent_dist.sample()
        latents = (latents * pipe.vae.config.scaling_factor).permute(0, 2, 1, 3, 4)       # (B, F, C, H, W)
        frames = pipe(
            prompt=prompt, negative_prompt=DAS_NEGATIVE_PROMPT,
            image=image.resize((DAS_W, DAS_H), Image.BILINEAR),
            num_videos_per_prompt=1, num_inference_steps=args.steps, num_frames=DAS_FRAMES,
            use_dynamic_cfg=True, guidance_scale=args.guidance_scale,
            generator=torch.Generator().manual_seed(args.seed),
            tracking_maps=latents, tracking_image=maps[:1], height=DAS_H, width=DAS_W,
        ).frames[0]
    del pipe
    free_gpu()
    return frames


# ──────────────────────────────────────────────────────────────────────────

def process_example(example_dir: Path, args) -> None:
    prefix = f"{args.run_name}_" if args.run_name else ""
    prediction = Path(args.prediction_root) / example_dir.name / f"{prefix}prediction.pt"
    inp = load_inputs(example_dir, prediction, args.init_frame)
    out_dir = Path(args.output) / example_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = lambda name: out_dir / f"{prefix}{name}"  # noqa: E731
    prompt = args.prompt or inp["caption"]
    image, K, pts2d, anchor = inp["image"], inp["K"], inp["points_2d"], inp["anchor_3d"]

    reproj = float(np.linalg.norm(project(anchor, K) - pts2d, axis=1).mean())
    print(f"  prediction {prediction} {inp['pred_3d'].shape}, prompt \"{prompt}\"; "
          f"query-point reprojection error {reproj:.2f} px")
    if reproj > 2.0:
        print(f"  WARNING: the intrinsics do not match the query points ({reproj:.1f} px); "
              f"the tracking video will be offset (see examples/data/davis_bmx_trees_das)")

    depth = da3_depth(image, args.da3_model, args.da3_process_res)
    W0, H0 = image.size
    u = np.clip(np.round(pts2d[:, 0]).astype(int), 0, W0 - 1)
    v = np.clip(np.round(pts2d[:, 1]).astype(int), 0, H0 - 1)
    ratios = anchor[:, 2] / depth[v, u]
    depth *= np.median(ratios)
    print(f"  DA3 depth scaled x{np.median(ratios):.2f} to the query points "
          f"(ratios {ratios.min():.2f}-{ratios.max():.2f})")

    extra = [list(map(float, p.split(","))) for p in args.mask_points.split(";") if p.strip()] \
        if args.mask_points else inp["mask_points"]
    mask_sam = sam3_mask(image, pts2d, extra)
    z_obj = float(np.median(anchor[:, 2]))
    mask = mask_sam & (np.abs(depth - z_obj) <= args.mask_depth_tol * z_obj)
    print(f"  SAM 3 mask {mask_sam.sum()} px ({len(extra)} extra click(s)) -> {mask.sum()} px "
          f"within {args.mask_depth_tol:.0%} of z={z_obj:.2f} m")
    overlay = np.asarray(image).astype(np.float32)
    overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array([255, 40, 40])
    Image.fromarray(draw_tracks(overlay.astype(np.uint8), np.r_[pts2d, np.array(extra).reshape(-1, 2)][:, None], 0)
                    ).save(out("mask_overlay.png"))

    tracking, overlays, info = build_tracking(inp, depth, mask, args)
    fps = args.model_fps * (DAS_FRAMES - 1) / info["future_steps"]      # the prediction in real time
    write_video(out("tracking.mp4"), tracking, fps)
    write_video(out("tracking_overlay.mp4"), overlays, fps)
    print(f"  tracking video: {DAS_FRAMES} frames, {info['grid_points']} grid points "
          f"({info['moving_points']} moving), query points move {info['query_motion_px']:.0f} px")

    result = [np.asarray(f) for f in das_generate(Path(args.checkpoint), image, prompt, tracking, args)]
    write_video(out("result.mp4"), result, fps)
    write_video(out("comparison.mp4"), [np.concatenate([a, b], axis=1) for a, b in zip(overlays, result)], fps)
    (out("das.json")).write_text(json.dumps({
        "example": str(example_dir), "prediction": str(prediction), "prompt": prompt,
        "input_frame": "init" if args.init_frame else "t0", "reprojection_error_px": reproj,
        "depth_scale": float(np.median(ratios)), "mask_px": [int(mask_sam.sum()), int(mask.sum())],
        "mask_points": extra, "fps": fps, **info,
        "das": {"checkpoint": str(args.checkpoint), "steps": args.steps,
                "guidance_scale": args.guidance_scale, "seed": args.seed},
    }, indent=2) + "\n")
    print(f"  wrote {out('result.mp4')}, {out('comparison.mp4')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", nargs="+", required=True, help="Example directories.")
    ap.add_argument("--run-name", default=None,
                    help="Prediction to use, <prediction-root>/<example>/<run-name>_prediction.pt "
                         "(as `run_molmo_motion.py --run-name`); also prefixes the outputs.")
    ap.add_argument("--prediction-root", default=str(_REPO_ROOT / "result" / "molmo_motion_prediction"))
    ap.add_argument("--output", default=str(_REPO_ROOT / "result" / "das"))
    ap.add_argument("--init-frame", action="store_true",
                    help="Start from the example's init frame (for `run_molmo_motion.py --init-frame` predictions).")
    ap.add_argument("--prompt", default=None, help="DaS prompt; default = the example's caption.")
    ap.add_argument("--mask-points", default=None,
                    help="Extra SAM clicks 'x,y;x,y' (input-frame pixels); default = the example's mask_points.json.")
    ap.add_argument("--mask-depth-tol", type=float, default=0.25,
                    help="Keep mask pixels within this fraction of the query points' median depth.")
    ap.add_argument("--fill-background", action=argparse.BooleanOptionalAction, default=True,
                    help="Static points behind the object fill the area it uncovers.")
    ap.add_argument("--density", type=int, default=70, help="Tracking grid size (DaS / SpaTracker: 70).")
    ap.add_argument("--model-fps", type=float, default=15.0, help="Frame rate of the prediction steps.")
    ap.add_argument("--da3-model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--da3-process-res", type=int, default=504)
    ap.add_argument("--checkpoint", default=str(_REPO_ROOT / "weights" / "Diffusion-As-Shader"))
    ap.add_argument("--steps", type=int, default=50, help="DaS denoising steps.")
    ap.add_argument("--guidance-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for i, name in enumerate(args.input, 1):
        example_dir = Path(name)
        print(f"[{i}/{len(args.input)}] {example_dir}")
        process_example(example_dir, args)


if __name__ == "__main__":
    main()
