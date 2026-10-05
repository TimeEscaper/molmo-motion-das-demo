#!/usr/bin/env python3
"""Render a pipeline run as an MP4: 2D tracks over the video (top) + 3D trajectories (bottom).

Reads from a finished video folder <work_dir>/<vid>/ (stages 3-5):
    video.*                                 the video the tracker saw
    tracks_2d.npz                           AllTracker 2D tracks, all points
    final_tracks/<vid>_filter_meta.npz      raw-lifted + smoothed 3D, keep mask
    camera/pose.npz                         camera-to-world poses used for the lift

Top panel: every query point with a short trail; points dropped by Stage 5 are drawn
hollow. Bottom panel, two 3D views in the world frame of the run (camera at the first
frame; +x right, +y down, +z forward -> plotted with depth into the screen, "up" up):
the scene with the estimated camera centre (black), and a zoom on the object. Both show
the smoothed kept tracks (solid) and the raw lifted points (gray dots), with equal axes.

Usage:
    python visualize_tracks.py --video_dir runs/my_run/pour_water_0001   # -> <video_dir>/viz.mp4
    python visualize_tracks.py --stack a.mp4 b.mp4 --labels ViPE static --out both.mp4
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TRAIL = 6  # frames of trail in the 2D panel


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def find_video(video_dir):
    hits = sorted(video_dir.glob("video.*"))
    if not hits:
        raise FileNotFoundError(f"no video.* in {video_dir}")
    return hits[0]


def find_pose(video_dir):
    p = video_dir / "camera" / "pose.npz"
    return p if p.exists() else None


def point_colors(n):
    cmap = plt.get_cmap("hsv")
    return [cmap(i / max(n, 1))[:3] for i in range(n)]


def draw_2d(frame, tracks, vis, keep, t, colors, title):
    img = frame.copy()
    h, w = img.shape[:2]
    for n in range(tracks.shape[1]):
        c = tuple(int(255 * v) for v in colors[n][::-1])  # RGB -> BGR
        t0 = max(0, t - TRAIL)
        for s in range(t0, t):
            if vis[s, n] and vis[s + 1, n]:
                p0 = tuple(np.round(tracks[s, n]).astype(int))
                p1 = tuple(np.round(tracks[s + 1, n]).astype(int))
                cv2.line(img, p0, p1, c, 1, cv2.LINE_AA)
        if vis[t, n]:
            p = tuple(np.round(tracks[t, n]).astype(int))
            cv2.circle(img, p, 4, c, -1 if keep[n] else 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, f"{title}  t={t}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def to_plot(p):
    """Camera/world (x right, y down, z forward) -> plot (x, z, -y): depth recedes, up is up."""
    return p[..., 0], p[..., 2], -p[..., 1]


def _cube(points, pad=1.15, min_half=0.02):
    pts = points.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1)]
    lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
    return (lo + hi) / 2, max((hi - lo).max() / 2, min_half) * pad


def make_3d_renderer(P_raw, P_smooth, vis, keep, cam_centers, colors, size_px, title):
    """Two 3D views side by side: the scene (camera + object) and a zoom on the object."""
    kept_pts = P_smooth[keep][vis[keep]] if keep.any() else P_raw[vis]
    obj_c, obj_h = _cube(kept_pts)
    scene_c, scene_h = _cube(np.concatenate([kept_pts.reshape(-1, 3), cam_centers]), min_half=0.1)
    W, H = size_px
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    T = P_smooth.shape[1]

    def draw(ax, t, ctr, half, name, show_cam):
        ax.cla()
        for n in np.where(keep)[0]:
            if not vis[n, : t + 1].any():
                continue
            ax.plot(*to_plot(P_smooth[n, : t + 1]), color=colors[n], lw=1.5)
            if vis[n, t]:
                ax.scatter(*to_plot(P_smooth[n, t]), color=colors[n], s=18, depthshade=False)
        raw = P_raw[:, t][vis[:, t]]
        if len(raw):
            ax.scatter(*to_plot(raw), color="gray", s=4, alpha=0.5, depthshade=False)
        if show_cam:
            ax.plot(*to_plot(cam_centers[: t + 1]), color="k", lw=1.5)
            ax.scatter(*to_plot(cam_centers[t]), color="k", marker="^", s=30, depthshade=False)
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[2] - half, ctr[2] + half)
        ax.set_zlim(-ctr[1] - half, -ctr[1] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x right [m]", fontsize=8)
        ax.set_ylabel("z depth [m]", fontsize=8)
        ax.set_zlabel("up [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=20, azim=-75 + 30 * t / max(T - 1, 1))
        ax.set_title(f"{name}  (cube side {200 * half:.0f} cm)", fontsize=9)

    def render(t):
        draw(axes[0], t, scene_c, scene_h, "scene: camera (black) + object", True)
        draw(axes[1], t, obj_c, obj_h, "object zoom", False)
        fig.suptitle(f"{title}  t={t}   smoothed kept tracks, gray = raw lift", fontsize=10)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return render


def write_mp4(path, frames, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def render_run(video_dir, out, title, fps):
    vid = video_dir.name
    frames, video_fps = read_video(find_video(video_dir))
    t2d = np.load(video_dir / "tracks_2d.npz")
    tracks, vis2d = t2d["tracks"], t2d["visibility"].astype(bool)  # (T, N, 2), (T, N)
    meta = np.load(video_dir / "final_tracks" / f"{vid}_filter_meta.npz")
    P_raw, P_smooth = meta["P_original"], meta["P_smoothed"]      # (N, T, 3)
    vis3d, keep = meta["visibility_all"].astype(bool), meta["keep_mask"].astype(bool)
    pose_path = find_pose(video_dir)
    T = min(len(frames), tracks.shape[0], P_smooth.shape[1])
    cams = (np.load(pose_path)["data"][:T, :3, 3] if pose_path is not None
            else np.zeros((T, 3)))
    if len(cams) < T:
        cams = np.concatenate([cams, np.repeat(cams[-1:], T - len(cams), 0)])

    colors = point_colors(tracks.shape[1])
    out_w = max(960, frames[0].shape[1])
    render3d = make_3d_renderer(P_raw, P_smooth, vis3d, keep, cams, colors,
                                (out_w, out_w // 2), title)
    out_frames = []
    for t in range(T):
        top = draw_2d(frames[t], tracks, vis2d, keep, t, colors,
                      f"{title}: {keep.sum()}/{len(keep)} tracks kept")
        top = cv2.resize(top, (out_w, round(top.shape[0] * out_w / top.shape[1])),
                         interpolation=cv2.INTER_LINEAR)
        out_frames.append(np.vstack([top, render3d(t)]))
    write_mp4(out, out_frames, fps or video_fps)
    print(f"wrote {out}  ({T} frames, {keep.sum()}/{len(keep)} tracks kept, pose={pose_path})")


def stack_videos(paths, labels, out, fps):
    vids = [read_video(p) for p in paths]
    T = min(len(f) for f, _ in vids)
    h = max(f[0].shape[0] for f, _ in vids)
    frames = []
    for t in range(T):
        row = []
        for (f, _), lab in zip(vids, labels or [None] * len(vids)):
            im = f[t]
            if im.shape[0] < h:
                im = np.vstack([im, np.full((h - im.shape[0], im.shape[1], 3), 255, np.uint8)])
            row.append(im)
        frames.append(np.hstack(row))
    write_mp4(out, frames, fps or vids[0][1])
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", type=Path, help="a video folder of a run, <work_dir>/<vid>")
    ap.add_argument("--title", default=None, help="label drawn on both panels (default: video id)")
    ap.add_argument("--stack", nargs="+", type=Path, help="instead: put these rendered MP4s side by side")
    ap.add_argument("--labels", nargs="+", help="unused, kept for symmetry with --stack")
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: source video fps)")
    ap.add_argument("--out", type=Path, default=None, help="default: <video_dir>/viz.mp4")
    args = ap.parse_args()
    if args.stack:
        if not args.out:
            ap.error("--stack needs --out")
        stack_videos(args.stack, args.labels, args.out, args.fps)
    else:
        if not args.video_dir:
            ap.error("--video_dir is required (or use --stack)")
        render_run(args.video_dir, args.out or args.video_dir / "viz.mp4",
                   args.title or args.video_dir.name, args.fps)


if __name__ == "__main__":
    main()
