# MolmoMotion × Diffusion as Shader demo

Forecast 3D point trajectories with **MolmoMotion** and turn them into videos with
**Diffusion as Shader (DaS)**, on simulated (MolmoSpaces), in-the-wild (YT-VIS, DAVIS) and
real-robot (ShareRobot) clips. Includes an annotation pipeline that produces MolmoMotion-style
3D tracks for robot episodes.

This repository is a fork of [allenai/molmo-motion](https://github.com/allenai/molmo-motion).
The upstream README (training, PointMotionBench evaluation, HF conversion, robotics) is kept as
[`README_original.md`](README_original.md); everything below describes what this fork adds.

## Contents

- [Project context](#project-context)
- [What this fork adds](#what-this-fork-adds)
- [Environment (uv)](#environment-uv)
- [Weights](#weights)
- [Data](#data)
- [Running the stages](#running-the-stages)
- [Notes and known limitations](#notes-and-known-limitations)

## Project context

| Component | What it is | Reference |
|---|---|---|
| **MolmoMotion** | 4B vision-language model that predicts the future 3D trajectories of query points from RGB history frames, the points' 2D/3D positions and a language instruction. We use the released **autoregressive** checkpoints: `H3-F30` (3 history frames, 30 future steps) and `H1-F32` (1 frame, 32 steps). The flow-matching variant of the paper is not released. | arXiv:2606.18558, [`README_original.md`](README_original.md) |
| **MolmoMotion-1M** | MolmoMotion's training corpus, annotated with the data-generation pipeline in [`data_generation/`](data_generation/). We sample test clips from its **MolmoSpaces** (simulation, fixed cameras) and **YT-VIS** (in the wild, hand-held cameras) subsets. | `allenai/molmo-motion-1m` |
| **ShareRobot** | Planning data of RoboBrain: 51k successful Open-X-Embodiment robot episodes, 30 frames each, with a goal and sub-steps. It has no 3D tracks, so we annotate episodes ourselves with the (adapted) data-generation pipeline. | arXiv:2502.21257 |
| **Diffusion as Shader (DaS)** | CogVideoX-5B image-to-video model conditioned on a "3D tracking video" (coloured 3D points of the first frame, moved and projected into every frame) and a text prompt. We build the tracking video from MolmoMotion's prediction. Code: [`das/`](das/) submodule. | arXiv:2501.03847 |
| **Depth Anything 3 (DA3)** | Metric depth, intrinsics and poses from images; a depth backend of the annotation pipeline and the source of the dense depth DaS needs. | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` |

```
ShareRobot archive ─ extract_sharerobot ─▶ run_pipeline (annotation) ─▶ sample_sharerobot ─┐
MolmoMotion-1M ───────────────────────────────── sample_molmospaces / sample_ytvis ────────┼─▶ run_molmo_motion ─▶ run_das
examples/data/davis_bmx_trees_das ─────────────────────────────────────────────────────────┘
```

## What this fork adds

### New scripts

| Script | Purpose |
|---|---|
| [`scripts/data/extract_sharerobot.py`](scripts/data/extract_sharerobot.py) | Export ShareRobot planning episodes (by name or at random) as `<video_id>.mp4` + `goal.txt` + `episode.json` (goal, sub-steps with frame ranges), streaming the ~320 GB archive once. |
| [`scripts/data/sample_molmospaces.py`](scripts/data/sample_molmospaces.py), [`scripts/data/sample_ytvis.py`](scripts/data/sample_ytvis.py) | Convert MolmoMotion-1M clips into MolmoMotion inputs: history frames, query points, 3D history, K, caption, ground-truth future, `clip.mp4`. |
| [`scripts/data/sample_sharerobot.py`](scripts/data/sample_sharerobot.py) | The same for annotated ShareRobot episodes, plus the episode's **init frame** (`frame_init.jpg`, points and K at it), to run the H=1 model from the start of the episode, before the gripper reaches the object. |
| [`scripts/run_molmo_motion.py`](scripts/run_molmo_motion.py) | Run MolmoMotion (H=1 or H=3) on examples: `prediction.pt`, 2D/3D renders and, with ground truth, ADE / FDE / PWT. `--init-frame` feeds the init frame instead of t0. |
| [`scripts/run_das.py`](scripts/run_das.py) | MolmoMotion prediction → DaS: DA3 depth (scaled to the query points), SAM 3 object mask (query points + optional extra clicks), sparse SpaTracker-style tracking video, DaS generation, plus the real clip for comparison. |
| [`data_generation/scripts/click_query_points.py`](data_generation/scripts/click_query_points.py) | Click query points by hand (needs a GUI OpenCV); `--episodes` writes them into the episode folders. |
| [`data_generation/scripts/visualize_tracks.py`](data_generation/scripts/visualize_tracks.py) | MP4 of an annotated video: 2D tracks over the video + 3D trajectories (also run automatically as stage 7). |

### Changes to the data-generation pipeline ([`data_generation/run_pipeline.py`](data_generation/run_pipeline.py))

- **Episode folders as input** (`--episodes`), as written by `extract_sharerobot.py`; the goal is the action.
- **Query points:** `--query_points auto` (default; Qwen3 → MolmoPoint → SAM 3 → K-means) or `manual` (clicked points from each episode's `query_points/`).
- **Fixed camera** (`static_camera: true`): depth and intrinsics are kept, poses become identity. With ViPE, pair it with `vipe_pipeline: static_camera` (frozen intrinsics, [`static_camera.yaml`](data_generation/third_party/vipe/configs/pipeline/static_camera.yaml)).
- **Depth backends** (`depth_backend: vipe | da3`): Depth Anything 3 as an alternative to ViPE; both write the same files.
- **Manipulated object only** (`keep_objects: first | moving | all`): grounding may name more objects than are moved (e.g. the destination). Objects that do not move — and with `first`, all but the first moving one in grounding order — are removed from every output.
- **One folder per video**, keeping only what is used, self-contained (`meta.json` with the action and the episode, a copy of the video, `config.yaml` of the run).
- **Stage 7: visualization** (`viz.mp4`) by default; `encode_480p: false` keeps the native resolution and frame rate.
- **New configs:** [`robot_static.yaml`](data_generation/configs/robot_static.yaml), [`sharerobot_static.yaml`](data_generation/configs/sharerobot_static.yaml) (ViPE), [`sharerobot_static_da3.yaml`](data_generation/configs/sharerobot_static_da3.yaml) (DA3).

Details: [`data_generation/README.md`](data_generation/README.md).

### Other changes

- [`examples/data/davis_bmx_trees_das/`](examples/data/davis_bmx_trees_das/): a copy of `davis_bmx_trees` with a **corrected K** and a mask hint.
  - The bundled principal point put the 3D points ~53 px off their 2D query points; the fix brings that to 0.3 px.
  - `mask_points.json` holds a click on the bike, which SAM segments separately from the rider.
- [`src/molmo_motion/numpy_compat.py`](src/molmo_motion/numpy_compat.py), and `np.bool` → `np.bool_` in `preprocessing/multimodal_collator.py`: the code and the MolmoMotion-1M NPZ pickles assume numpy 2, while the venv needs numpy 1.26 (DA3).
- [`das/`](das/): DiffusionAsShader as a git submodule (only its CogVideoX tracking model is used).
- [`pyproject.toml`](pyproject.toml) / `uv.lock`: one uv environment for everything (below).

## Environment (uv)

Everything runs in **one** project venv (`.venv/`, Python 3.12, PyTorch 2.9.1 + CUDA 12.8), managed with [uv](https://docs.astral.sh/uv/): MolmoMotion, the data-generation pipeline (ViPE, SAM 3, AllTracker, MolmoPoint), Depth Anything 3 and DaS.

```bash
git clone --recurse-submodules <this repo> molmo-motion-das-demo
cd molmo-motion-das-demo            # (for an existing clone: git submodule update --init)
uv sync                             # creates .venv from uv.lock
source .venv/bin/activate           # or prefix commands with .venv/bin/python
```

**Requirements:**
- Linux and an NVIDIA GPU. We used 80 GB; DaS alone needs ~32 GB.
- CUDA 12.x `nvcc` on `PATH`: ViPE builds a CUDA extension during `uv sync`. It is installed editable and without build isolation, so it finds its configs.

**Pins and why** (`pyproject.toml`):

| Pin | Why |
|---|---|
| `numpy==1.26.0` | Depth Anything 3 requires `numpy<2` |
| `transformers==4.57.1` | MolmoPoint/Molmo2 grounding and ViPE's GroundingDINO break on 5.x |
| `diffusers==0.33.1`, `sentencepiece` | DaS (CogVideoX tracking pipeline) |
| `gdown<6` | ViPE's checkpoint download uses `fuzzy=` |
| `depth-anything-3` | installed from a pinned GitHub source archive, because cloning through our proxy fails |
| `addict` | an undeclared Depth Anything 3 dependency |

**Notes:**
- **SAM 3** is imported from [`data_generation/third_party/sam3`](data_generation/third_party/sam3); the scripts put it on `sys.path`, since the installed copy is incomplete.
- **GUI tools:** the venv ships `opencv-python-headless`, so `click_query_points.py` needs a GUI OpenCV build and a display.

**Environment variables:** create `.env` in the repo root (read by the data scripts):

```bash
MOLMO_MOTION_1M_ROOT=/path/to/molmo-motion-1m
SHAREROBOT_ROOT=/path/to/ShareRobot/ShareRobot
```

Optionally set `HF_HOME` / `TORCH_HOME` for the model caches.

## Weights

The scripts load the checkpoints from two places by default:
- **MolmoMotion:** `weights/` (gitignored).
- **DaS:** `das/checkpoints/`, the DaS repo's own convention (ignored by the submodule's git).

Download them, or link existing copies:

```bash
mkdir -p weights
hf download allenai/MolmoMotion-4B-H3-F30 --local-dir weights/MolmoMotion-4B-H3-F30
hf download allenai/MolmoMotion-4B-H1-F32 --local-dir weights/MolmoMotion-4B-H1-F32
hf download EXCAI/Diffusion-As-Shader     --local-dir das/checkpoints/Diffusion-As-Shader   # ~25 GB
# or link existing copies, e.g.
# ln -s /mnt/vol1/shared/weights/molmo-motion/MolmoMotion-4B-H3-F30 weights/MolmoMotion-4B-H3-F30
# ln -s /path/to/DiffusionAsShader/checkpoints das/checkpoints
```

`run_das.py --checkpoint` points DaS elsewhere.

These download automatically into the Hugging Face / torch caches on first use:

| Model | Used by | Note |
|---|---|---|
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | annotation (`depth_backend: da3`), `run_das.py` | CC BY-NC 4.0 weights |
| `facebook/sam3` | annotation stage 1, `run_das.py` | **gated**: request access on Hugging Face, then `hf auth login` (or set `HF_TOKEN`) |
| `allenai/MolmoPoint-Vid-4B`, `allenai/Molmo2-8B`, `Qwen/Qwen3-0.6B` | annotation stage 1 (automatic query points) | ~56 GB |
| ViPE priors, AllTracker | annotation stages 2–3 | torch hub |

## Data

| Data | Needed for | Preparation |
|---|---|---|
| MolmoMotion-1M (`molmospaces/`, `ytvis/`) | `sample_molmospaces.py`, `sample_ytvis.py` | `hf download allenai/molmo-motion-1m --repo-type dataset --local-dir $MOLMO_MOTION_1M_ROOT`, then rebuild the videos with each subset's reconstruction script (see [`README_original.md`](README_original.md#downloading-the-dataset-and-benchmark)) |
| ShareRobot | `extract_sharerobot.py` | `hf download BAAI/ShareRobot --repo-type dataset --local-dir <dir>`; set `SHAREROBOT_ROOT` to the folder containing `planning/`, which holds the `train/` and `test/` JSONs and `images/rt_frames_success.tar.gz.part.*` (~320 GB) |
| DAVIS BMX | `run_molmo_motion.py`, `run_das.py` | bundled: [`examples/data/davis_bmx_trees_das/`](examples/data/davis_bmx_trees_das/) |

## Running the stages

All commands run from the repository root with the venv active; the annotation pipeline runs from `data_generation/`. Outputs go to `result/` (gitignored):

```
result/
├── sharerobot_raw/<episode>/                                1. extracted ShareRobot episodes
├── sharerobot_annotated/<video_id>/                         2. annotated episodes (3D tracks, camera, depth, viz)
├── molmo_motion_input/<dataset>/<example>/                  3. MolmoMotion inputs
├── molmo_motion_prediction/<dataset>/<setting>/<example>/   4. predictions (+ metrics)
└── das/<dataset>/<setting>/<example>/                       5. DaS videos
```

### 1. Extract ShareRobot episodes

```bash
python scripts/data/extract_sharerobot.py --video bridge_3715 bridge_7119 bridge_18068
# or random ones from one source dataset:
python scripts/data/extract_sharerobot.py --num_videos 5 --dataset bridge --seed 1
```

- **Output:** `result/sharerobot_raw/<episode>/{<video_id>.mp4, goal.txt, episode.json}`: 30 frames at 4 fps.
- **Speed:** bridge episodes come first in the archive and are found quickly; a full pass over the archive takes ~25 min.

### 2. Annotate them (3D tracks)

```bash
cd data_generation
# Depth Anything 3 + fixed camera (used for our results)
python run_pipeline.py --episodes ../result/sharerobot_raw \
    --config configs/sharerobot_static_da3.yaml --work_dir ../result/sharerobot_annotated
# ViPE + fixed camera instead
python run_pipeline.py --episodes ../result/sharerobot_raw \
    --config configs/sharerobot_static.yaml --work_dir ../result/sharerobot_annotated_vipe
cd ..
```

- **Output, per video** (`<work_dir>/<video_id>/`): `meta.json`, `video.mp4`, `query_points/`, `depth.zip`, `camera/{intrinsics,pose}.npz`, `tracks_2d.npz`, `tracks_3d.npz`, `final_tracks/`, `clips.json`, `viz.mp4`.
- **Speed:** ~1.5 min per episode once the grounding models are loaded.
- **Caching:** every stage caches its output; `--start_stage N` re-runs from a given stage.

Manual query points instead of automatic grounding:

```bash
python data_generation/scripts/click_query_points.py --episodes result/sharerobot_raw   # needs a GUI
cd data_generation && python run_pipeline.py --episodes ../result/sharerobot_raw \
    --config configs/sharerobot_static_da3.yaml --query_points manual --work_dir ../result/sharerobot_manual
```

### 3. Build MolmoMotion inputs

```bash
python scripts/data/sample_molmospaces.py --video pick_place_2cam_randomized__house_717__00000004__exo_camera_1 --stride 2
python scripts/data/sample_molmospaces.py --video pick_place_color_5cam__house_6031__00000013__droid_shoulder_light_randomization
python scripts/data/sample_ytvis.py --video aef3e2cb0e --offset 8
python scripts/data/sample_ytvis.py --video b673e7dcfb --offset 10
python scripts/data/sample_sharerobot.py --num_videos 3 --data_root result/sharerobot_annotated
```

- **Output:** each writes `result/molmo_motion_input/<dataset>/<dataset>_<video>/`.
- **Random sampling:** all three scripts take `--num_videos N --seed S`.
- **ShareRobot options:** `sample_sharerobot.py` also takes `--init_frame` (default 0), `--caption goal|step`, `--t0` and `--tag`.

### 4. Run MolmoMotion

One folder per dataset and setting: `h1`, `h3`, and for ShareRobot also `h1_init`.

```bash
for ds in molmospaces ytvis sharerobot; do
  for h in 1 3; do
    python scripts/run_molmo_motion.py --input result/molmo_motion_input/$ds --history $h \
        --output result/molmo_motion_prediction/$ds/h$h
  done
done
python scripts/run_molmo_motion.py --input result/molmo_motion_input/sharerobot --history 1 --init-frame \
    --output result/molmo_motion_prediction/sharerobot/h1_init
# DAVIS BMX (bundled example, no ground truth)
for h in 1 3; do
  python scripts/run_molmo_motion.py --input examples/data/davis_bmx_trees_das --history $h \
      --output result/molmo_motion_prediction/davis/h$h
done
```

- **Output, per example:** `prediction.pt` (P × F × 3, camera frame at t0, metres), `2d.gif`, `3d.png`. With ground truth, also `side_by_side.gif` and `metrics.json` (ADE / FDE / PWT).
- **Inputs:** `--input` takes example folders, or a folder of examples.
- **Naming:** `--run-name` prefixes the files instead of using separate folders.
- **Speed:** each call loads the model once (~2 min).

### 5. Generate videos with DaS

```bash
for ds in molmospaces ytvis sharerobot; do
  for s in h1 h3; do
    python scripts/run_das.py --input result/molmo_motion_input/$ds \
        --prediction-root result/molmo_motion_prediction/$ds/$s --output result/das/$ds/$s
  done
done
python scripts/run_das.py --input result/molmo_motion_input/sharerobot --init-frame \
    --prediction-root result/molmo_motion_prediction/sharerobot/h1_init --output result/das/sharerobot/h1_init
for s in h1 h3; do
  python scripts/run_das.py --input examples/data/davis_bmx_trees_das \
      --prediction-root result/molmo_motion_prediction/davis/$s --output result/das/davis/$s
done
```

Output, per example:

| File | Contents |
|---|---|
| `tracking.mp4` | the DaS conditioning video |
| `tracking_overlay.mp4` | the tracking video over the input frame, with the query tracks |
| `mask_overlay.png` | the object mask on the input frame |
| `result.mp4` | 49 generated frames, re-timed to the prediction |
| `comparison.mp4` | tracking overlay \| result |
| `original.mp4` | the real clip, where the example has one |
| `das.json` | parameters and diagnostics |

- **Speed:** about 6 min per example on an A100; the DaS pipeline is loaded once per call.
- **Useful options:**
  - `--prompt` (default: the example's caption, i.e. the ShareRobot goal);
  - `--mask-points "x,y;..."` (extra SAM clicks; default: the example's `mask_points.json`);
  - `--no-fill-background`, `--steps`, `--seed`.

## Notes and known limitations

- **DaS assumes a fixed camera:** the background of the tracking video never moves. For hand-held clips (YT-VIS), the generated video shows the object's predicted motion in front of a static background.
- **The robot arm** is not among the moved points in ShareRobot clips; only the manipulated object is. DaS, trained on human and Mixamo videos, tends to erase or replace the arm. The next things to try are adding the gripper with `--mask-points`, or a prompt that mentions the robot.
- **Small objects** get few tiles in the sparse 70 × 70 tracking grid (e.g. ~45 for the croissant), which weakens the motion signal for DaS. The sparse grid is used anyway: the dense per-pixel rendering made DaS pan the camera, or paint a "ghost" object into the uncovered area.
- **Intrinsics must match the query points:** `run_das.py` reports the reprojection error and warns above 2 px. The bundled `davis_bmx_trees` example has a wrong principal point; use `davis_bmx_trees_das`.
- **Frame rates:** ShareRobot episodes are ~4 fps, while MolmoMotion's steps are treated as 15 fps when re-timing the DaS output; `original.mp4` keeps the source rate.
- **Determinism:** MolmoMotion decodes greedily, so identical inputs give identical predictions. Annotation (grounding, ViPE / DA3) varies slightly between runs, so re-annotated episodes can select different query points.
