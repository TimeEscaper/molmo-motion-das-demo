"""Export ShareRobot planning episodes as MP4 + goal.txt + episode.json (no statistics, no figures).

ShareRobot (RoboBrain, arXiv:2502.21257) planning episodes are 30 frames subsampled from
successful Open-X-Embodiment episodes. Their frames only ship inside a ~320 GB split tar.gz
(`planning/images/rt_frames_success.tar.gz.part.*`), and the goal / sub-steps have to be
reconstructed from the 1M planning QA pairs (`planning/{train,test}/*.json`): the QA pairs of an
episode give its goal, its atomic sub-steps and, through the number of frames each QA sees,
the frame where each sub-step starts.

Episodes are chosen up front -- by name (`--video`) or uniformly at random from the planning
JSONs (`--num_videos`, optionally only from one source `--dataset`) -- and collected in one
sequential pass over the archive (~230 MB/s, i.e. up to ~25 min for the whole archive; the
archive is grouped by source shard, bridge first). Episodes already exported are not streamed
again. Writes, per episode:

    <out_root>/<episode dir>/<video_id>.mp4     the 30 frames, clean (no captions), H.264
    <out_root>/<episode dir>/goal.txt           the episode goal (task instruction)
    <out_root>/<episode dir>/episode.json       goal, sub-steps with frame ranges, fps, ...

`<episode dir>` is the episode key with "/" -> "__" and "#" -> "_" (e.g.
`rtx_frames_success_14__49_bridge_episode_3715`) and `<video_id>` is `<dataset>_<episode>`
(e.g. `bridge_3715`), the names scripts/data/sample_sharerobot.py and the data-generation
pipeline use. Odd frame sizes are cropped by one pixel (H.264 needs even sizes).

A `--video` name may be the episode key (`rtx_frames_success_14/49_bridge#episode_3715`), the
episode dir name, or the video id (`bridge_3715`, if unique).

Usage:
    python scripts/data/extract_sharerobot.py --num_videos 5 --dataset bridge --seed 1
    python scripts/data/extract_sharerobot.py --video bridge_3715 bridge_379
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

from dotenv import load_dotenv

import imageio.v2 as imageio
import numpy as np
from PIL import Image

# Repo uses a src layout (`src/molmo_motion`). This script lives in
# scripts/data/, so the repo root is two levels up. Load `.env` before any
# SHAREROBOT_ROOT lookup.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

DEFAULT_ROOT = Path(os.environ["SHAREROBOT_ROOT"])
DEFAULT_OUT = REPO_ROOT / "result" / "sharerobot_raw"

NUM_FRAMES = 30  # every ShareRobot episode is subsampled to 30 frames

# QA whose image prefix ends where sub-step `selected_step` starts / is finished.
STEP_START_TASKS = {
    "Planning_Task", "Planning_with_Context_Task", "Planning_Remaining_Steps_Task",
    "Generative_Affordance_Task", "Future_Prediction_Task", "Success_(Negative)_Task",
    "Discriminative_Affordance_(Positive)_Task",
}
STEP_END_TASKS = {"Past_Description_Task", "Success_(Positive)_Task"}

# Native rate of each OXE source and its mean native episode length (frames); ShareRobot keeps
# no timestamps, so this only estimates the effective rate of the 30 sampled frames.
SOURCE_FPS = {
    "robo_set": (5, None), "bridge": (5, 35.6), "fmb": (10, 132.1), "dobbe": (3.75, 218.9),
    "jaco_play": (10, 71.9), "berkeley_autolab_ur5": (5, 97.9),
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": (5, 50.0),
    "nyu_door_opening_surprising_effectiveness": (3, 42.2), "plex_robosuite": (20, None),
    "aloha_mobile": (50, 1827.0), "qut_dexterous_manpulation": (30, None),
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": (5, 136.3),
    "asu_table_top_converted_externally_to_rlds": (5, 237.4),
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": (10, 73.4),
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": (5, 144.0),
    "dlr_edan_shared_control_converted_externally_to_rlds": (5, 85.8),
    "conq_hose_manipulation": (30, 59.5), "dlr_sara_pour_converted_externally_to_rlds": (5, 129.7),
    "viola": (20, 510.5), "dlr_sara_grid_clamp_converted_externally_to_rlds": (5, 71.2),
    "ucsd_kitchen_dataset_converted_externally_to_rlds": (5, 26.5),
    "utokyo_xarm_bimanual_converted_externally_to_rlds": (10, 21.6), "cmu_stretch": (5, 185.3),
}


# ──────────────────────────────────────────────────────────────────────────
# Episodes from the planning QA.
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    key: str
    goal: str = ""
    steps: dict[int, str] = field(default_factory=dict)       # step index -> text
    starts: dict[int, set] = field(default_factory=dict)      # step index -> first frame(s)

    @property
    def dataset(self) -> str:
        return re.sub(r"^\d+_", "", self.key.split("/")[-1].split("#")[0])

    @property
    def video_id(self) -> str:
        return f"{self.dataset}_{self.key.rsplit('#episode_', 1)[-1]}"

    @property
    def dir_name(self) -> str:
        return self.key.replace("/", "__").replace("#", "_")

    def segments(self) -> list[tuple[int, str, int | None, int | None]]:
        """(k, text, first frame, end frame exclusive); None where no QA pins the boundary."""
        out = []
        last = max(self.steps) if self.steps else -1
        for k in sorted(self.steps):
            start = min(self.starts[k]) if k in self.starts else None
            end = min(self.starts[k + 1]) if k + 1 in self.starts else (NUM_FRAMES if k == last else None)
            out.append((k, self.steps[k], start, end))
        return out


def episode_key(path: str) -> str:
    """'…/rt_frames_success/rtx_frames_success_14/49_bridge#episode_4263[/frame_3.png]'
    -> 'rtx_frames_success_14/49_bridge#episode_4263'."""
    key = path.split("rt_frames_success/")[-1].strip("/")
    return key.rsplit("/", 1)[0] if key.endswith(".png") else key


def _load_planning_file(path: str) -> list[tuple]:
    rows = []
    for it in json.loads(Path(path).read_text()):
        conv = it["conversations"]
        question = conv[0]["value"].split("\n", 1)[-1]  # drop the "<image> <image> ..." prefix
        rows.append((episode_key(it["id"]), it["task"], it["selected_step"], len(it["image"]),
                     question, conv[1]["value"]))
    return rows


def load_episodes(root: Path) -> dict[str, Episode]:
    # planning/jsons/ is exactly train ∪ test, so it is not loaded.
    files = sorted(str(p) for s in ("train", "test") for p in (root / "planning" / s).glob("*.json"))
    with Pool(min(len(files), os.cpu_count() or 1)) as pool:
        chunks = pool.map(_load_planning_file, files)
    eps: dict[str, Episode] = {}
    for key, task, k, n_frames, question, answer in (row for chunk in chunks for row in chunk):
        ep = eps.get(key) or eps.setdefault(key, Episode(key, starts={0: {0}}))
        if task in STEP_START_TASKS:
            ep.starts.setdefault(k, set()).add(n_frames)
        elif task in STEP_END_TASKS:
            ep.starts.setdefault(k + 1, set()).add(n_frames)
        # Sub-step texts: numbered lists "1-<…>, 2-<…>" and single-step answers.
        for m in re.finditer(r"(\d+)-<([^>]*)>", question + " " + answer):
            ep.steps.setdefault(int(m.group(1)) - 1, m.group(2))
        if task in ("Planning_Task", "Generative_Affordance_Task", "Past_Description_Task"):
            ep.steps.setdefault(k, answer.strip("<>"))
        elif task in ("Success_(Positive)_Task", "Success_(Negative)_Task"):
            ep.steps.setdefault(k, re.search(r"<([^>]*)>", question).group(1))
        if task == "Planning_Task" and not ep.goal:
            ep.goal = re.search(r"<([^>]*)>", question).group(1)
    return eps


def resolve(name: str, eps: dict[str, Episode]) -> Episode:
    """Episode by key, dir name or (unique) video id."""
    if name in eps:
        return eps[name]
    hits = [ep for ep in eps.values() if name in (ep.dir_name, ep.video_id)]
    if len(hits) != 1:
        raise SystemExit(f"--video {name!r}: {'no' if not hits else len(hits)} matching episodes"
                         + (f" ({', '.join(ep.key for ep in hits)}); pass the full key" if hits else ""))
    return hits[0]


# ──────────────────────────────────────────────────────────────────────────
# Frames from the archive.
# ──────────────────────────────────────────────────────────────────────────

def stream_episodes(root: Path, wanted: set[str], scan_gb: float | None):
    """Yield (key, [frame arrays in frame order]) for every wanted episode, in archive order,
    reading the split tar.gz sequentially until all are found (or `scan_gb` GB are read)."""
    parts = sorted((root / "planning" / "images").glob("rt_frames_success.tar.gz.part.*"))
    if not parts:
        raise SystemExit(f"no archive parts under {root / 'planning' / 'images'}")
    decomp = "pigz" if shutil.which("pigz") else "gzip"
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    unz = subprocess.Popen([decomp, "-dc"], stdin=cat.stdout, stdout=subprocess.PIPE)
    cat.stdout.close()
    left, read, cur, frames = set(wanted), 0, None, {}
    try:
        with tarfile.open(fileobj=unz.stdout, mode="r|") as tar:
            for m in tar:
                read += m.size
                if not (m.isfile() and m.name.endswith(".png")):
                    continue
                key = episode_key(m.name)
                if key != cur:                          # the archive is grouped by episode
                    if cur in left:
                        left.discard(cur)
                        yield cur, [frames[i] for i in sorted(frames)], read
                    if not left or (scan_gb and read > scan_gb * 1e9):
                        break
                    cur, frames = key, {}
                if key in left:
                    i = int(re.search(r"frame_(\d+)\.png$", m.name).group(1))
                    frames[i] = np.asarray(Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB"))
            else:
                if cur in left:
                    left.discard(cur)
                    yield cur, [frames[i] for i in sorted(frames)], read
    finally:
        unz.kill()
        cat.kill()
        unz.wait()
        cat.wait()


def export(ep: Episode, frames: list[np.ndarray], out_dir: Path, fps: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    frames = [f[:h - h % 2, :w - w % 2] for f in frames]       # H.264 / yuv420p needs even sizes
    video = out_dir / f"{ep.video_id}.mp4"
    imageio.mimsave(video, frames, fps=fps, codec="libx264", macro_block_size=1, quality=9)
    native_fps, mean_len = SOURCE_FPS.get(ep.dataset, (None, None))
    meta = {
        "episode": ep.key,
        "dataset": ep.dataset,
        "video_id": ep.video_id,
        "video": video.name,
        "video_fps": fps,
        "num_frames": len(frames),
        "image_size_wh": [int(frames[0].shape[1]), int(frames[0].shape[0])],
        "goal": ep.goal,
        "steps": [{"step": k + 1, "text": t, "start": s, "end": e} for k, t, s, e in ep.segments()],
        "steps_note": f"frames [start, end) of the {NUM_FRAMES}-frame episode; null where no QA pins the boundary",
        "native_fps": native_fps,
        "effective_fps_estimate": round(NUM_FRAMES * native_fps / mean_len, 2) if native_fps and mean_len else None,
    }
    (out_dir / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "goal.txt").write_text(ep.goal.strip() + "\n")
    return video


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", nargs="+", help="Episode key, episode dir name or video id (e.g. bridge_3715).")
    src.add_argument("--num_videos", type=int, help="Sample this many episodes uniformly at random.")
    ap.add_argument("--dataset", default=None, help="With --num_videos: only this OXE source (e.g. bridge).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="ShareRobot root (or $SHAREROBOT_ROOT).")
    ap.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fps", type=float, default=4.0,
                    help="Playback rate of the MP4 (ShareRobot keeps no timestamps; bridge is ~4 fps).")
    ap.add_argument("--scan_gb", type=float, default=None,
                    help="Stop after this many uncompressed GB of the archive (default: until all found).")
    ap.add_argument("--overwrite", action="store_true", help="Re-export episodes that already exist.")
    args = ap.parse_args()
    if args.dataset and not args.num_videos:
        ap.error("--dataset only applies to --num_videos")

    print(f"[load] planning QA from {args.root / 'planning'} ...")
    eps = {k: ep for k, ep in load_episodes(args.root).items() if ep.goal}
    print(f"[load] {len(eps):,} episodes with a goal")

    if args.video:
        chosen = list(dict.fromkeys(resolve(name, eps).key for name in args.video))
    else:
        pool = sorted(k for k, ep in eps.items() if not args.dataset or ep.dataset == args.dataset)
        if not pool:
            raise SystemExit(f"no episodes of dataset {args.dataset!r}")
        chosen = random.Random(args.seed).sample(pool, min(args.num_videos, len(pool)))

    done = lambda ep: all((args.out_root / ep.dir_name / f).exists()  # noqa: E731
                          for f in (f"{ep.video_id}.mp4", "episode.json", "goal.txt"))
    todo = {k for k in chosen if args.overwrite or not done(eps[k])}
    for k in chosen:
        if k not in todo:
            print(f"[skip] {eps[k].video_id}: already in {args.out_root / eps[k].dir_name}")
    if todo:
        print(f"[stream] {len(todo)} episode(s) from the archive (up to ~25 min for a full pass) ...")
        for key, frames, read in stream_episodes(args.root, todo, args.scan_gb):
            ep = eps[key]
            video = export(ep, frames, args.out_root / ep.dir_name, args.fps)
            todo.discard(key)
            warn = "" if len(frames) == NUM_FRAMES else f"  [warning: {len(frames)} frames]"
            print(f"[ok] {ep.video_id}: \"{ep.goal}\", {len(ep.steps)} sub-steps -> {video} "
                  f"({read / 1e9:.1f} GB scanned){warn}")
    for k in sorted(todo):
        print(f"[missing] {eps[k].video_id} ({k}) not found in the scanned part of the archive")
    print(f"exported {len(chosen) - len(todo)}/{len(chosen)} episodes into {args.out_root}")


if __name__ == "__main__":
    main()
