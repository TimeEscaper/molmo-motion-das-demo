"""MolmoMotion quickstart: predict a 3D point trajectory and render it as a GIF.

Single end-to-end script:

  1. Load the model + processor (once, shared by all examples).
  2. For every requested example, build one inference input from the bundled
     clip (history RGB frames, the query points' 2D pixel coords at t_0, their
     3D camera-frame history, and an action caption).
  3. ``out = model.predict_trajectory(**inputs)`` -> ``out.future_3d`` is a
     ``(P, F, 3)`` tensor of absolute camera-frame XYZ (meters).
  4. ``render_trajectory_animation(out.future_3d, ...)`` projects that
     trajectory back onto the static t_0 frame and animates it as a growing 2D
     track, coloured with the ``magma`` gradient (one polyline per point, a
     bright dot at the moving end). The visualization is taken *directly* from
     ``out`` -- no intermediate files.
  5. ``visualize_trajectory_3d(points_3d_history, out.future_3d, ...)`` plots
     the past and predicted points in the 3D camera frame and saves it as an
     image; with ground truth, the GT trajectory gets its own 3D panel next to
     the prediction (same axis limits).
  6. With ground truth (``gt_future_3d.pt`` + ``clip.mp4``, written by the
     ``scripts/data/sample_*.py`` converters), also render prediction |
     ground truth | original video side by side. On by default; silently
     skipped for examples without ground truth.
  7. With ground truth, compute ADE / FDE / PWT as defined in the paper
     (Appendix C.4) and save them as JSON.

Run (needs a GPU for the 4B model)::

    pip install -e ".[viz]"
    python scripts/run_molmo_motion.py

Several examples in one run (names under examples/data/ or paths; a directory
without meta.json is expanded to all example sub-directories inside it)::

    python scripts/run_molmo_motion.py --input davis_bmx_trees davis_flamingo new/ytvis_29f2332d30
    python scripts/run_molmo_motion.py --input new
    python scripts/run_molmo_motion.py --input 'new/ytvis_*' 'molmospaces_*_stride2'

Use the single-frame model (H=1, F=32; conditioned only on the t_0 frame and
the t_0 3D points)::

    python scripts/run_molmo_motion.py --history 1

No GPU? Render the visualizations from the bundled released-model prediction
instead -- this exercises the exact same visualization paths on identical
arrays::

    python scripts/run_molmo_motion.py --from-prediction

Output, in ``<output>/<input folder name>/`` (``--output`` defaults to ``result/molmo_motion_prediction/``):
``prediction.pt`` (the ``(P, F, 3)`` prediction), ``2d.gif``, ``3d.png`` and,
with ground truth, ``side_by_side.gif`` + ``metrics.json``. ``--run-name NAME`` prefixes the files (``NAME_2d.gif``, ...);
``--video-format mp4`` writes MP4s instead.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Repo uses a src layout (`src/molmo_motion`). Put it on the path so
# `python scripts/run_molmo_motion.py` finds the package without an install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

EXAMPLES_DIR = Path(__file__).parent / "data"


# ──────────────────────────────────────────────────────────────────────────
# Visualization:
#   1. Project (P, F, 3) -> image plane and animate a magma 2D track.
#   2. Render past and predicted 3D points in the camera frame as an image.
# Depends only on the ``[viz]`` extra (matplotlib + imageio[ffmpeg]).
# ──────────────────────────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def project_camera_xyz_to_pixel(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole-project (..., 3) camera-frame XYZ (meters) to (..., 2) pixels."""
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = K[0, 0] * (xyz[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("GIF export needs ffmpeg (pip install imageio[ffmpeg])")
        return exe


def save_animation(frames: list[np.ndarray], output_path, fps: float) -> str:
    """Write RGB ``frames`` to ``output_path``; the format follows the suffix.

    ``.gif`` goes through ffmpeg's two-pass palettegen/paletteuse (one 256-colour
    palette optimised over the whole animation, sierra dithering), which keeps
    the image and the magma gradient smooth without per-frame palette flicker.
    GIF delays are whole centiseconds, so e.g. 30 fps is encoded as a 3/3/4 cs
    pattern that averages to exactly 30 fps. Anything else is written as an
    h264 MP4.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".gif":
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps, codec="libx264", macro_block_size=1,
                        ffmpeg_params=["-pix_fmt", "yuv420p"])
        return str(path)

    h, w = frames[0].shape[:2]
    cmd = [
        _ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
        "-filter_complex",
        "split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
        "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
        "-loop", "0", str(path),
    ]
    raw = np.ascontiguousarray(np.stack(frames), dtype=np.uint8).tobytes()
    subprocess.run(cmd, input=raw, check=True)
    return str(path)


def render_trajectory_animation(
    future_3d,
    *,
    t0_image: Image.Image,
    intrinsics,
    points_2d_at_t0,
    output_path: str,
    fps: int = 30,
    seconds: float = 3.6,
    cmap_name: str = "magma",
    gradient_floor: float = 0.15,
    gradient_top: float = 0.88,
    line_width: float = 2.4,
    dot_size: float = 36.0,
    pad: bool = False,
    pad_margin: float = 0.05,
) -> str:
    """Animate a predicted 3D trajectory as a 2D track over the static t_0 frame.

    Each of the ``P`` points gets one polyline that grows from its t_0 query
    pixel along the projected future path. The polyline is vertex-coloured by
    cumulative arc length through a colormap sub-range (oldest = dark, newest =
    bright), with a filled dot at the moving end.

    Args:
        future_3d: ``(P, F, 3)`` camera-frame XYZ in meters -- exactly
            ``model.predict_trajectory(...).future_3d``. Tensor or ndarray.
        t0_image: the t_0 RGB frame the track is drawn over (kept static).
        intrinsics: ``(3, 3)`` camera matrix ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``.
        points_2d_at_t0: ``(P, 2)`` query-point pixel coords at t_0 -- the
            anchor each polyline grows out of.
        output_path: where to write the animation (``.gif`` or ``.mp4``).
        fps, seconds: frame rate and total duration of the reveal.
        cmap_name: matplotlib colormap (default ``magma``).
        gradient_floor, gradient_top: map arc length into this colormap
            sub-range so the oldest end is not pure black and the newest end is
            not washed out.
        line_width, dot_size: trail thickness and moving-dot area (pt^2).
        pad: if True, extend the canvas (filled black) so trail portions that
            project outside the image stay visible (e.g. the bmx clip rides off
            the right edge). If False, the canvas is the image and off-frame
            track is clipped.
        pad_margin: extra border around the trajectory when ``pad`` is on,
            as a fraction of the image's larger side.

    Returns:
        ``output_path``.
    """
    full = _track_pixels(future_3d, intrinsics, points_2d_at_t0)   # (P, T, 2)
    img = np.asarray(t0_image.convert("RGB"))
    extent = _canvas_extent([full], img.shape[1], img.shape[0], pad, pad_margin)
    n_frames = max(2, int(round(fps * seconds)))
    frames = _trajectory_frames(
        full, img, extent, n_frames, n_steps=full.shape[1] - 1,
        cmap_name=cmap_name, gradient_floor=gradient_floor, gradient_top=gradient_top,
        line_width=line_width, dot_size=dot_size,
    )
    return save_animation(frames, output_path, fps)


render_trajectory_mp4 = render_trajectory_animation


def _track_pixels(future_3d, intrinsics, points_2d_at_t0) -> np.ndarray:
    """``(P, 1 + F, 2)`` pixel track: the t_0 query pixel followed by the
    projected future, so each trail starts on the tracked point."""
    pred = _to_numpy(future_3d).astype(np.float64)                 # (P, F, 3)
    K = _to_numpy(intrinsics).astype(np.float64)                   # (3, 3)
    anchor_2d = _to_numpy(points_2d_at_t0).astype(np.float64)      # (P, 2)
    px = project_camera_xyz_to_pixel(pred, K)                      # (P, F, 2)
    return np.concatenate([anchor_2d[:, None, :], px], axis=1)


def _canvas_extent(tracks, W, H, pad, pad_margin):
    """Canvas ``(x0, x1, y0, y1)``: the image by default, or grown to fit every
    track in ``tracks`` (plus a margin) when padding is requested."""
    if not pad:
        return 0.0, float(W), 0.0, float(H)
    m = pad_margin * max(W, H)
    xs = np.concatenate([t[..., 0].ravel() for t in tracks])
    ys = np.concatenate([t[..., 1].ravel() for t in tracks])
    return (min(0.0, float(xs.min()) - m), max(float(W), float(xs.max()) + m),
            min(0.0, float(ys.min()) - m), max(float(H), float(ys.max()) + m))


def _canvas_size(extent):
    """Output ``(width, height)`` for ``extent``, rounded to even pixels for yuv420p."""
    x0, x1, y0, y1 = extent
    return int(round(x1 - x0)) // 2 * 2, int(round(y1 - y0)) // 2 * 2


def _trajectory_frames(
    full: np.ndarray,
    img: np.ndarray,
    extent,
    n_frames: int,
    n_steps: int,
    *,
    cmap_name: str = "magma",
    gradient_floor: float = 0.15,
    gradient_top: float = 0.88,
    line_width: float = 2.4,
    dot_size: float = 36.0,
) -> list[np.ndarray]:
    """Render the growing-trail animation of ``full`` (``(P, T, 2)`` pixels) over
    the static ``img``. The playhead sweeps ``[0, n_steps]`` future steps over
    ``n_frames`` frames; a track shorter than ``n_steps`` stays at its end."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    P, T, _ = full.shape
    H, W = img.shape[:2]

    # Per-point gradient parametrised by cumulative arc length (stable as the
    # trail grows), remapped into [floor, top] of the colormap.
    seg_len = np.linalg.norm(np.diff(full, axis=1), axis=2)        # (P, T-1)
    arclen = np.concatenate([np.zeros((P, 1)), np.cumsum(seg_len, axis=1)], axis=1)
    arcnorm = arclen / np.clip(arclen[:, -1:], 1e-6, None)         # (P, T) in [0,1]
    cvals = gradient_floor + (gradient_top - gradient_floor) * arcnorm

    cmap = matplotlib.colormaps[cmap_name]
    dpi = 100
    x0, x1, y0, y1 = extent
    out_w, out_h = _canvas_size(extent)

    frames = []
    for k in range(n_frames):
        head = min((k / (n_frames - 1)) * n_steps, T - 1)   # fractional playhead in [0, T-1]
        ni = int(np.floor(head))
        fig = plt.figure(figsize=((x1 - x0) / dpi, (y1 - y0) / dpi), dpi=dpi,
                         facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_facecolor("black")
        ax.imshow(img, extent=[0, W, H, 0])            # image at its native pixels
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)                            # image coords: y grows down

        for p in range(P):
            if head < 1.0:                             # before the first step: just the dot
                tip = full[p, 0] + (full[p, 1] - full[p, 0]) * head
            else:
                pts = full[p, : ni + 1]
                nxt = min(ni + 1, T - 1)
                tip = full[p, ni] + (full[p, nxt] - full[p, ni]) * (head - ni)
                line_pts = np.concatenate([pts, tip[None, :]], axis=0)
                line_c = np.concatenate([cvals[p, : ni + 1], [cvals[p, nxt]]])
                segs = np.stack([line_pts[:-1], line_pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1), zorder=4)
                lc.set_array(0.5 * (line_c[:-1] + line_c[1:]))
                lc.set_linewidth(line_width)
                lc.set_capstyle("round")
                ax.add_collection(lc)
            ax.scatter([tip[0]], [tip[1]], s=dot_size, color=cmap(0.95),
                       zorder=6, edgecolors="white", linewidths=0.4)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        # Force exact even dimensions so yuv420p / h264 is happy.
        frame = np.asarray(Image.fromarray(buf).resize((out_w, out_h)))
        frames.append(frame)
        plt.close(fig)
    return frames


def _video_panel(frame: np.ndarray, extent) -> np.ndarray:
    """Place a raw video frame on the same (possibly padded) canvas as the
    trajectory panels so all panels share pixel geometry."""
    x0, x1, y0, y1 = extent
    H, W = frame.shape[:2]
    left, top = int(round(-x0)), int(round(-y0))
    canvas = np.zeros((int(round(y1 - y0)), int(round(x1 - x0)), 3), dtype=np.uint8)
    canvas[top:top + H, left:left + W] = frame
    return np.asarray(Image.fromarray(canvas).resize(_canvas_size(extent)))


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray(frame)
    try:
        font = ImageFont.load_default(size=max(14, frame.shape[0] // 22))
    except TypeError:  # Pillow < 10.1 has no sized default font
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((8, 6), text, fill="white", font=font,
                             stroke_width=2, stroke_fill="black")
    return np.asarray(img)


def render_side_by_side_animation(
    pred_3d,
    gt_3d,
    clip_frames,
    t0_index: int,
    *,
    t0_image: Image.Image,
    intrinsics,
    points_2d_at_t0,
    output_path: str,
    fps: int = 30,
    seconds: float = 3.6,
    pad: bool = False,
    pad_margin: float = 0.05,
    gap: int = 8,
) -> str:
    """Three synchronized panels in one row: predicted trail over the static
    t_0 frame, ground-truth trail over the same frame, and the original video
    from t_0 to the end of the horizon.

    Args:
        pred_3d, gt_3d: ``(P, F, 3)`` camera-frame-at-t_0 XYZ (meters); F may
            differ (the longer one sets the timeline; the shorter one holds its
            last position).
        clip_frames: ``(N, H, W, 3)`` uint8 video frames, one per model step.
        t0_index: index of the t_0 frame in ``clip_frames``; panel 3 shows
            ``clip_frames[t0_index + step]`` at playhead step ``step``.
        Other args: as in :func:`render_trajectory_animation`.
    """
    img = np.asarray(t0_image.convert("RGB"))
    pred_px = _track_pixels(pred_3d, intrinsics, points_2d_at_t0)
    gt_px = _track_pixels(gt_3d, intrinsics, points_2d_at_t0)
    extent = _canvas_extent([pred_px, gt_px], img.shape[1], img.shape[0], pad, pad_margin)
    n_frames = max(2, int(round(fps * seconds)))
    n_steps = max(pred_px.shape[1], gt_px.shape[1]) - 1

    pred_frames = _trajectory_frames(pred_px, img, extent, n_frames, n_steps)
    gt_frames = _trajectory_frames(gt_px, img, extent, n_frames, n_steps)

    clip_frames = np.asarray(clip_frames)
    last_step = len(clip_frames) - 1 - t0_index
    spacer = np.zeros((pred_frames[0].shape[0], gap, 3), dtype=np.uint8)
    frames = []
    for k in range(n_frames):
        step = int(round((k / (n_frames - 1)) * n_steps))
        video = _video_panel(clip_frames[t0_index + min(step, last_step)], extent)
        frames.append(np.concatenate([
            _label(pred_frames[k], "Prediction"), spacer,
            _label(gt_frames[k], "Ground truth"), spacer,
            _label(video, f"Video  t0+{min(step, last_step)}"),
        ], axis=1))
    return save_animation(frames, output_path, fps)


render_side_by_side_mp4 = render_side_by_side_animation


def visualize_trajectory_3d(
    points_3d_history=None,
    future_3d=None,
    output_path: str = "trajectory_3d.png",
    *,
    gt_future_3d=None,
    gt_future_vis=None,
    past_3d=None,
    predicted_3d=None,
    history_3d=None,
    title: str | None = None,
    frame: str = "camera",
    figsize: tuple[float, float] | None = None,
    elev: float = 25.0,
    azim: float = -60.0,
    cmap_name: str = "tab10",
    show_legend: bool = True,
    dpi: int = 150,
) -> str:
    """Visualize past and predicted 3D points in the camera frame and save as an image.

    Plots the 3D trajectory of each query point in camera coordinates (meters):
      - Past points (t <= 0): dashed line with markers for historical steps.
      - Query anchor (t = 0): prominent circle marker where prediction begins.
      - Predicted future (t > 0): solid line extending along the predicted path.
      - Predicted endpoint (t = F): triangle marker at the final predicted step.

    If ``gt_future_3d`` is given, the ground-truth future is drawn in a second
    3D panel to the right of the prediction (same history, same axis limits),
    so the two can be compared directly.

    Args:
        points_3d_history: ``(H, P, 3)`` or ``(P, H, 3)`` camera-frame history
            (meters). Tensor or ndarray.
        future_3d: ``(P, F, 3)`` camera-frame predicted future (meters).
            Tensor or ndarray.
        output_path: Where to save the output image (PNG/JPG).
        gt_future_3d: Optional ``(P, F_gt, 3)`` ground-truth future in the same
            frame; ``F_gt`` may differ from ``F``.
        gt_future_vis: Optional ``(P, F_gt)`` bool; invisible GT steps are not
            drawn (their stored coordinates may be zero-filled NaNs).
        past_3d, history_3d: Optional aliases for ``points_3d_history``.
        predicted_3d: Optional alias for ``future_3d``.
        title: Optional custom figure title.
        frame: Coordinate frame convention:
            - ``"camera"`` (default): X right, Y down, Z forward/depth.
            - ``"upright"``: X right, Z forward, -Y up.
        figsize: Matplotlib figure size (width, height) in inches; defaults
            to ``(11, 8.5)``, or ``(20, 8.5)`` with the ground-truth panel.
        elev, azim: 3D perspective elevation and azimuth angles in degrees.
        cmap_name: Matplotlib colormap for distinct point colors (default ``"tab10"``).
        show_legend: Whether to display point and trajectory style legends.
        dpi: Saved image resolution in dots per inch.

    Returns:
        ``output_path``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if points_3d_history is None:
        points_3d_history = past_3d if past_3d is not None else history_3d
    if future_3d is None:
        future_3d = predicted_3d

    if points_3d_history is None and future_3d is None:
        raise ValueError("At least one of points_3d_history or future_3d must be provided.")

    h = _to_numpy(points_3d_history)
    f = _to_numpy(future_3d)

    # Auto-detect if posargs were passed in reverse order (future, past)
    if h is not None and f is not None:
        if h.ndim == 3 and f.ndim == 3:
            if h.shape[1] > 10 and f.shape[0] <= 5:
                h, f = f, h
            elif h.shape[0] > 10 and f.shape[1] <= 5 and h.shape[1] == f.shape[0]:
                h, f = f, h

    # Standardize future to (P, F, 3)
    if f is not None:
        f = f.astype(np.float64)
        if f.ndim != 3 or f.shape[-1] != 3:
            raise ValueError(f"future_3d must have shape (P, F, 3); got {f.shape}")
        P = f.shape[0]
        F = f.shape[1]
    else:
        P = None
        F = 0

    # Standardize history to (P, H, 3)
    if h is not None:
        h = h.astype(np.float64)
        if h.ndim != 3 or h.shape[-1] != 3:
            raise ValueError(f"points_3d_history must have shape (H, P, 3) or (P, H, 3); got {h.shape}")
        if P is not None and h.shape[1] == P and h.shape[0] != P:
            # (H, P, 3) -> (P, H, 3)
            h = np.swapaxes(h, 0, 1)
        elif P is None:
            if h.shape[0] <= 5 and h.shape[1] > h.shape[0]:
                h = np.swapaxes(h, 0, 1)
            P = h.shape[0]
        H = h.shape[1]
    else:
        H = 0

    # Standardize ground truth to (P, F_gt, 3), NaN where not visible
    g = _to_numpy(gt_future_3d)
    if g is not None:
        g = g.astype(np.float64).copy()
        if g.ndim != 3 or g.shape[-1] != 3 or g.shape[0] != P:
            raise ValueError(f"gt_future_3d must have shape ({P}, F_gt, 3); got {g.shape}")
        if gt_future_vis is not None:
            g[~_to_numpy(gt_future_vis).astype(bool)] = np.nan

    def _transform(pts: np.ndarray):
        if frame == "upright":
            # X -> right, Z -> forward, -Y -> up
            return pts[..., 0], pts[..., 2], -pts[..., 1]
        elif frame == "camera":
            # Camera frame: X right, Y down, Z forward/depth
            return pts[..., 0], pts[..., 1], pts[..., 2]
        else:
            raise ValueError(f"Unknown frame convention: {frame!r}. Use 'camera' or 'upright'.")

    # (panel title, future array, legend word)
    panels = [("Prediction", f, "Predicted")]
    if g is not None:
        panels.append(("Ground truth", g, "Ground-truth"))
    if figsize is None:
        figsize = (11.0, 8.5) if len(panels) == 1 else (20.0, 8.5)

    # Shared limits so both cubes use the same scale.
    all_pts = np.concatenate([np.stack(_transform(a), axis=-1).reshape(-1, 3)
                              for a in (h, f, g) if a is not None])
    lo, hi = np.nanmin(all_pts, axis=0), np.nanmax(all_pts, axis=0)
    margin = 0.05 * np.maximum(hi - lo, 1e-3)
    lo, hi = lo - margin, hi + margin

    cmap = plt.colormaps.get(cmap_name, plt.colormaps["tab10"])
    fig = plt.figure(figsize=figsize)
    for i, (panel_title, fut, word) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        _draw_3d_panel(ax, h, fut, P, cmap, _transform)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])

        if frame == "upright":
            ax.set_xlabel("Camera X (right, m)", fontsize=10, labelpad=8)
            ax.set_ylabel("Camera Z (forward, m)", fontsize=10, labelpad=8)
            ax.set_zlabel("-Y (up, m)", fontsize=10, labelpad=8)
        else:
            ax.set_xlabel("Camera X (right, m)", fontsize=10, labelpad=8)
            ax.set_ylabel("Camera Y (down, m)", fontsize=10, labelpad=8)
            ax.set_zlabel("Camera Z (depth / forward, m)", fontsize=10, labelpad=8)
        ax.grid(True, linestyle=":", alpha=0.6)

        n_fut = 0 if fut is None else fut.shape[1]
        if len(panels) > 1:
            ax.set_title(f"{panel_title} ({n_fut} future steps)", fontsize=12, pad=15)
        else:
            if title is None:
                if h is not None and f is not None:
                    title = f"3D Camera-Frame Trajectories ({P} points, {H} past + {F} future steps)"
                elif f is not None:
                    title = f"3D Predicted Trajectories ({P} points, {F} future steps)"
                else:
                    title = f"3D Past Trajectories ({P} points, {H} past steps)"
            ax.set_title(title, fontsize=12, pad=15)

        if show_legend:
            style_handles = []
            if h is not None:
                style_handles.append(Line2D([0], [0], color="gray", linestyle="--", lw=1.8, label="Past trajectory (t ≤ 0)"))
                style_handles.append(Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray", markeredgecolor="black", markersize=7, lw=0, label="Query anchor (t = 0)"))
            if fut is not None:
                style_handles.append(Line2D([0], [0], color="gray", linestyle="-", lw=2.2, label=f"{word} future (t > 0)"))
                style_handles.append(Line2D([0], [0], marker="^", color="gray", markerfacecolor="gray", markeredgecolor="black", markersize=7, lw=0, label=f"{word} endpoint (t = F)"))
            if i == 0:
                point_handles = [
                    Line2D([0], [0], color=cmap(p % cmap.N), lw=2, label=f"Point {p}")
                    for p in range(P)
                ]
                leg1 = ax.legend(handles=point_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
                                 fontsize=8, title="Tracked Points", ncol=2 if P > 8 else 1)
                ax.add_artist(leg1)
            ax.legend(handles=style_handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
                      fontsize=8, title="Trajectory Legend")

    if len(panels) > 1:
        if title is None:
            title = f"3D Camera-Frame Trajectories ({P} points, {H} past steps)"
        fig.suptitle(title, fontsize=13)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _draw_3d_panel(ax, h, fut, P, cmap, transform):
    """Draw history ``h`` ``(P, H, 3)`` and future ``fut`` ``(P, F, 3)`` (either
    may be None; NaN future steps leave gaps) into one 3D axis."""
    for p in range(P):
        color = cmap(p % cmap.N)
        if h is not None:
            hx, hy, hz = transform(h[p])
            ax.plot(hx, hy, hz, linestyle="--", color=color, alpha=0.75, linewidth=1.8)
            if h.shape[1] > 1:
                ax.scatter(hx[:-1], hy[:-1], hz[:-1], color=color, marker="o", s=25, alpha=0.6)
            # Query anchor at t_0
            ax.scatter([hx[-1]], [hy[-1]], [hz[-1]], color=color, marker="o", s=60,
                       edgecolors="black", linewidths=1.2, zorder=5)

        if fut is None:
            continue
        valid = np.flatnonzero(np.isfinite(fut[p]).all(axis=-1))
        if len(valid) == 0:
            continue
        first, last = valid[0], valid[-1]

        if h is not None:
            bridge = np.stack([h[p, -1], fut[p, first]], axis=0)
            bx, by, bz = transform(bridge)
            ax.plot(bx, by, bz, linestyle=":", color=color, alpha=0.6, linewidth=1.5)

        fx, fy, fz = transform(fut[p])
        ax.plot(fx, fy, fz, linestyle="-", color=color, alpha=0.9, linewidth=2.2)
        mid = valid[valid < last]
        if len(mid):
            ax.scatter(fx[mid], fy[mid], fz[mid], color=color, marker=".", s=15, alpha=0.4)
        # Final future point at t_F
        ax.scatter([fx[last]], [fy[last]], [fz[last]], color=color, marker="^", s=65,
                   edgecolors="black", linewidths=0.8, zorder=6)


render_trajectory_3d = visualize_trajectory_3d


# ──────────────────────────────────────────────────────────────────────────
# Inputs.
# ──────────────────────────────────────────────────────────────────────────

def load_example(example_dir: Path, history_size: int, init_frame: bool = False):
    """Load the bundled clip's frames, query points, history, intrinsics, action. With
    `init_frame` (H=1), the input is the example's init frame instead of t0 (frame_init.jpg,
    points_*_at_init.pt, intrinsics_K_init.pt, as written by scripts/data/sample_sharerobot.py);
    the ground-truth future is unchanged."""
    meta = json.loads((example_dir / "meta.json").read_text())
    caption_file = example_dir / "caption.txt"
    action = caption_file.read_text().strip() if caption_file.exists() else meta["action"]
    if init_frame:
        if not (example_dir / "frame_init.jpg").exists():
            raise SkipExample(f"no frame_init.jpg in {example_dir}")
        return (meta, [Image.open(example_dir / "frame_init.jpg").convert("RGB")],
                torch.load(example_dir / "points_2d_at_init.pt"),         # (P, 2)
                torch.load(example_dir / "points_3d_at_init.pt"),         # (1, P, 3)
                torch.load(example_dir / "intrinsics_K_init.pt"), action)
    history_frames = [
        Image.open(example_dir / f"frame_t{i:+d}.jpg").convert("RGB")
        for i in range(-(history_size - 1), 1)        # H=3 -> t-2, t-1, t+0
    ]
    points_2d_at_t0 = torch.load(example_dir / "points_2d_at_t0.pt")     # (P, 2)
    # Bundled tensor ships with 3 history frames; slice the last `history_size`
    # so the H=1 model gets just t_0 (shape (1, P, 3)) and the H=3 model gets
    # t-2..t_0 (shape (3, P, 3)). Same indexing as `history_frames` above.
    points_3d_history = torch.load(example_dir / "points_3d_history.pt")[-history_size:]
    intrinsics = torch.load(example_dir / "intrinsics_K.pt")             # (3, 3)
    return meta, history_frames, points_2d_at_t0, points_3d_history, intrinsics, action


def load_ground_truth(example_dir: Path, meta: dict, *, with_clip: bool) -> dict | None:
    """Ground truth as written by the ``scripts/data/sample_*.py`` converters,
    or None if the example has none (e.g. the bundled DAVIS/EgoDex clips).

    Keys: ``future_3d`` ``(P, F, 3)``, ``future_vis`` ``(P, F)`` bool or None,
    and -- when ``with_clip`` and ``clip.mp4`` exists -- ``clip`` ``(N, H, W, 3)``
    uint8 plus ``t0_index``, the index of t_0 within it.
    """
    gt_path = example_dir / "gt_future_3d.pt"
    if not gt_path.exists():
        return None
    vis_path = example_dir / "gt_future_vis.pt"
    gt = {
        "future_3d": torch.load(gt_path),
        "future_vis": torch.load(vis_path) if vis_path.exists() else None,
    }
    clip_path = example_dir / "clip.mp4"
    if with_clip and clip_path.exists() and "clip_mp4_frame_indices" in meta:
        import imageio.v2 as imageio
        gt["clip"] = np.stack(imageio.mimread(clip_path, memtest=False))
        gt["t0_index"] = meta["clip_mp4_frame_indices"].index(meta["t0_absolute"])
    return gt


def prediction_from_jsonl(jsonl_path: Path, video: str) -> torch.Tensor:
    """Load the released-model ``(P, F, 3)`` prediction for ``video`` from the
    eval JSONL (``pred_raw_combined``) -- identical in shape/units to
    ``out.future_3d``."""
    for line in jsonl_path.read_text().splitlines():
        row = json.loads(line)
        if row["video"] == video:
            return torch.tensor(row["pred_raw_combined"], dtype=torch.float32)
    raise SkipExample(f"no bundled prediction for video={video!r} in {jsonl_path.name}")


class SkipExample(Exception):
    pass


# ──────────────────────────────────────────────────────────────────────────
# Metrics.
# ──────────────────────────────────────────────────────────────────────────

PWT_THRESHOLDS_M = (0.01, 0.02, 0.05, 0.10, 0.20)


def compute_trajectory_metrics(pred_3d, gt_3d, gt_vis=None) -> dict:
    """ADE / FDE / PWT exactly as in the paper (Appendix C.4), in meters.

    Scoring uses the horizon shared by prediction and ground truth
    (``min(F_pred, F_gt)`` steps) and only GT-visible (point, step) pairs; the
    query points are visible at t_0 by construction, so ``gt_vis`` is the whole
    evaluation mask. FDE is taken at the last step of that horizon, over the
    points visible there. PWT is the mean over ``PWT_THRESHOLDS_M`` of the
    fraction of scored pairs with error strictly below the threshold.
    Undefined values (nothing visible) are None.
    """
    pred = _to_numpy(pred_3d).astype(np.float64)
    gt = _to_numpy(gt_3d).astype(np.float64)
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"point count mismatch: pred {pred.shape}, gt {gt.shape}")
    F = min(pred.shape[1], gt.shape[1])
    pred, gt = pred[:, :F], gt[:, :F]
    vis = (np.ones(gt.shape[:2], dtype=bool) if gt_vis is None
           else _to_numpy(gt_vis).astype(bool)[:, :F])
    vis &= np.isfinite(gt).all(-1) & np.isfinite(pred).all(-1)

    err = np.linalg.norm(pred - gt, axis=-1)                       # (P, F)
    scored = err[vis]
    final = err[vis[:, -1], -1]
    pwt = {f"{t:g}": (float((scored < t).mean()) if scored.size else None)
           for t in PWT_THRESHOLDS_M}
    return {
        "ADE": float(scored.mean()) if scored.size else None,
        "FDE": float(final.mean()) if final.size else None,
        "PWT": float(np.mean(list(pwt.values()))) if scored.size else None,
        "PWT_per_threshold": pwt,
        "units": "m",
        "n_points": int(pred.shape[0]),
        "horizon_pred": int(_to_numpy(pred_3d).shape[1]),
        "horizon_gt": int(_to_numpy(gt_3d).shape[1]),
        "horizon_scored": int(F),
        "n_scored": int(scored.size),
        "n_points_final": int(final.size),
    }


def resolve_examples(names: list[str]) -> list[Path]:
    """Map ``--input`` entries to example directories.

    Each entry is a path or a name under ``examples/data/``. A directory with a
    ``meta.json`` is one example; any other directory is expanded to its
    immediate sub-directories that are examples. Entries with wildcards
    (``*``, ``?``, ``[...]``, e.g. ``new/*`` or ``new/ytvis_*``) are matched
    relative to the working directory, falling back to ``examples/data/``;
    matches that are not example directories are ignored.
    """
    dirs: list[Path] = []
    for name in names:
        if any(c in name for c in "*?["):
            matches = sorted(glob.glob(name)) or sorted(glob.glob(str(EXAMPLES_DIR / name)))
            found = [Path(m) for m in matches if (Path(m) / "meta.json").exists()]
            if not found:
                raise SystemExit(f"pattern {name!r} matches no example directories "
                                 f"(relative to the working directory or {EXAMPLES_DIR})")
            dirs.extend(found)
            continue
        path = Path(name)
        if path.is_file():
            continue                     # e.g. a shell-expanded new/* that also hit loose files
        if not path.is_dir():
            path = EXAMPLES_DIR / name
        if not path.is_dir():
            raise SystemExit(f"example {name!r} not found (neither a directory nor under {EXAMPLES_DIR})")
        if (path / "meta.json").exists():
            dirs.append(path)
            continue
        found = sorted(d for d in path.iterdir() if (d / "meta.json").exists())
        if not found:
            raise SystemExit(f"{path} has no meta.json and no example sub-directories")
        dirs.extend(found)
    return list(dict.fromkeys(d.resolve() for d in dirs))


WEIGHTS_DIR = _REPO_ROOT / "weights"
# History size -> (released checkpoint, its future horizon).
RELEASED_MODELS = {
    3: (str(WEIGHTS_DIR / "MolmoMotion-4B-H3-F30"), 30),
    1: (str(WEIGHTS_DIR / "MolmoMotion-4B-H1-F32"), 32),
}


def load_predictor(model_path: str, history: int, future_horizon: int):
    """Load the model once; return ``predict(history_frames, points_2d_at_t0,
    points_3d_history, action) -> (P, F, 3)`` float32 CPU tensor."""
    from molmo_motion import MolmoMotion, MolmoMotionProcessor

    processor = MolmoMotionProcessor.from_pretrained(model_path)
    H = processor.config.history_size
    if H != history:
        raise SystemExit(f"{model_path} is an H={H} checkpoint but --history={history}.")
    model = MolmoMotion.from_pretrained(model_path)
    model._internal = model._internal.to(torch.bfloat16).cuda()  # 4B params

    def predict(history_frames, points_2d_at_t0, points_3d_history, action) -> torch.Tensor:
        inputs = processor(
            history_frames=history_frames,
            points_2d_at_t0=points_2d_at_t0,
            points_3d_history=points_3d_history,
            action=action,
            future_horizon=future_horizon,
        )
        inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.predict_trajectory(**inputs)
        return out.future_3d.float().cpu()          # (P, F, 3), camera-frame meters

    return predict


def process_example(example_dir: Path, args, predict) -> list[str]:
    """Predict (or load the bundled prediction) and render every requested
    visualization into ``<output>/<input folder name>/``. Returns the written paths."""
    (meta, history_frames, points_2d_at_t0, points_3d_history,
     intrinsics, action) = load_example(example_dir, args.history, args.init_frame)
    t0_image = history_frames[-1]

    if predict is None:
        # No model: pull the shipped (P, F, 3) prediction. Same array the live
        # model would hand back as out.future_3d.
        future_3d = prediction_from_jsonl(EXAMPLES_DIR / "predictions_h3.jsonl", meta["video"])
        print(f"  loaded bundled prediction: future_3d {tuple(future_3d.shape)}")
    else:
        future_3d = predict(history_frames, points_2d_at_t0, points_3d_history, action)
        print(f"  predicted future_3d {tuple(future_3d.shape)}  "
              f"(expect ({points_2d_at_t0.shape[0]}, {args.future_horizon}, 3))")

    gt = load_ground_truth(example_dir, meta, with_clip=args.side_by_side)
    out_dir = Path(args.output) / example_dir.name
    prefix = f"{args.run_name}_" if args.run_name else ""
    out_path = lambda name: str(out_dir / f"{prefix}{name}")  # noqa: E731
    ext = f".{args.video_format}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(future_3d, out_path("prediction.pt"))       # (P, F, 3), e.g. for scripts/molmo_to_das.py
    written = [out_path("prediction.pt")]

    # 0. ADE / FDE / PWT against the ground truth, when there is one.
    if gt is not None:
        metrics = {
            "example": example_dir.name,
            "run_name": args.run_name,
            "history": args.history,
            "input_frame": "init" if args.init_frame else "t0",
            "prediction_source": "predictions_h3.jsonl" if predict is None else args.model,
            **compute_trajectory_metrics(future_3d, gt["future_3d"], gt["future_vis"]),
        }
        metrics_path = Path(out_path("metrics.json"))
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        written.append(str(metrics_path))
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"  ADE={fmt(metrics['ADE'])} m  FDE={fmt(metrics['FDE'])} m  PWT={fmt(metrics['PWT'])}  "
              f"({metrics['horizon_scored']} steps scored)")

    # 1. 2D overlay animation over the static frame.
    if not args.skip_2d:
        written.append(render_trajectory_animation(
            future_3d,
            t0_image=t0_image,
            intrinsics=intrinsics,
            points_2d_at_t0=points_2d_at_t0,
            output_path=out_path(f"2d{ext}"),
            pad=args.pad,
        ))

    # 2. 3D point trajectories in the camera frame; GT in its own panel.
    if not args.skip_3d:
        written.append(visualize_trajectory_3d(
            points_3d_history=points_3d_history,
            future_3d=future_3d,
            gt_future_3d=gt["future_3d"] if gt else None,
            gt_future_vis=gt["future_vis"] if gt else None,
            output_path=out_path("3d.png"),
        ))

    # 3. Prediction | ground truth | original video, synchronized in one row.
    if args.side_by_side and gt is not None and "clip" in gt:
        written.append(render_side_by_side_animation(
            future_3d,
            gt["future_3d"],
            gt["clip"],
            gt["t0_index"],
            t0_image=t0_image,
            intrinsics=intrinsics,
            points_2d_at_t0=points_2d_at_t0,
            output_path=out_path(f"side_by_side{ext}"),
            pad=args.pad,
        ))

    for path in written:
        print(f"  wrote {path}")
    return written


def _cleanup():
    """Release per-example figures, frames and cached GPU memory."""
    import matplotlib.pyplot as plt
    plt.close("all")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", nargs="+", default=["davis_bmx_trees"],
                    help="One or more inputs: sub-directories of examples/data/ or paths. "
                         "A directory without meta.json expands to all examples inside it; "
                         "wildcards like 'new/*' or 'new/ytvis_*' are resolved against the "
                         "working directory, then examples/data/.")
    ap.add_argument("--output", default="result/molmo_motion_prediction",
                    help="Directory for results. Each input is written to "
                         "<output>/<input folder name>/ (default: result/molmo_motion_prediction/).")
    ap.add_argument("--history", type=int, choices=sorted(RELEASED_MODELS), default=3,
                    help="History size H: 3 = H3-F30 model (t-2, t-1, t0), "
                         "1 = H1-F32 model (only the t0 frame and its 3D points).")
    ap.add_argument("--model", default=None,
                    help="Checkpoint path; defaults to the released model for --history.")
    ap.add_argument("--future-horizon", type=int, default=None,
                    help="Frames to predict; defaults to 30 for H=3 and 32 for H=1.")
    ap.add_argument("--run-name", default=None,
                    help="Optional prefix of the files written to <output>/<input folder name>/, "
                         "e.g. to keep outputs of different checkpoints apart.")
    ap.add_argument("--video-format", choices=["gif", "mp4"], default="gif",
                    help="Format of the 2D and side-by-side animations.")
    ap.add_argument("--pad", action="store_true",
                    help="Extend the canvas (black border) so track that "
                         "projects off the frame stays visible.")
    ap.add_argument("--skip-2d", action="store_true", help="Skip rendering the 2D animation.")
    ap.add_argument("--skip-3d", action="store_true", help="Skip rendering the 3D image.")
    ap.add_argument("--side-by-side", action=argparse.BooleanOptionalAction, default=True,
                    help="Render prediction | ground truth | original video in one row; "
                         "skipped for examples without gt_future_3d.pt + clip.mp4.")
    ap.add_argument("--init-frame", action="store_true",
                    help="H=1 only: feed the example's init frame (frame_init.jpg + points at it) "
                         "instead of t0, still scoring the future of t0.")
    ap.add_argument("--from-prediction", action="store_true",
                    help="Skip the model (no GPU needed); use the bundled "
                         "released-model prediction from predictions_h3.jsonl.")
    args = ap.parse_args()

    default_model, default_horizon = RELEASED_MODELS[args.history]
    model_path = args.model = args.model or default_model
    args.future_horizon = args.future_horizon or default_horizon
    if args.init_frame and args.history != 1:
        raise SystemExit("--init-frame needs --history 1")
    if args.from_prediction and args.history != 3:
        raise SystemExit("--from-prediction only has bundled H=3 predictions "
                         "(predictions_h3.jsonl); drop --history 1.")

    example_dirs = resolve_examples(args.input)
    predict = None if args.from_prediction else load_predictor(
        model_path, args.history, args.future_horizon)

    failed = []
    for i, example_dir in enumerate(example_dirs, 1):
        print(f"[{i}/{len(example_dirs)}] {example_dir}")
        try:
            process_example(example_dir, args, predict)
        except SkipExample as e:
            print(f"  [skip] {e}")
            failed.append(example_dir.name)
        except Exception:
            traceback.print_exc()
            failed.append(example_dir.name)
        finally:
            _cleanup()

    print(f"done: {len(example_dirs) - len(failed)}/{len(example_dirs)} examples")
    if failed:
        print("failed/skipped: " + ", ".join(failed))
        if len(failed) == len(example_dirs):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
