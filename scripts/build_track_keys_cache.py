"""Pre-scan molmo-motion-1m tracks NPZs to record which object keys each
video actually exposes. The upstream `*_split.json` for some entries
references keys (e.g. `obj1`) that don't exist in the corresponding
`_3d.npz` (upstream re-clustered the tracks but didn't update the split).
We use this cache to filter at Trajectory3DDataset init time so training
never hits a KeyError mid-step.

Run this once after downloading/reconstructing the corpus, before
training (Trajectory3DDataset raises a FileNotFoundError pointing here
if the cache is missing):

    export MOLMO_MOTION_1M_ROOT=/your/path/to/molmo-motion-1m
    python scripts/build_track_keys_cache.py

Output: one JSON per dataset at
  $MOLMO_MOTION_1M_TRACK_KEYS_CACHE (default:
  $HOME/.cache/molmo_motion_1m_track_keys)/{dataset}.json
mapping each file_id (or slug) → list of available object keys.

For datasets whose NPZs are flat (single object per file) the key list
is just ["__flat__"] and the dataset loader treats every entry's
clips_by_object key as present.
"""

import argparse
import json
import os
import runpy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Track NPZs hold numpy-2 pickles and the venv has numpy 1.26: alias numpy._core (run the
# module file, not `import molmo_motion...`, which would load the whole package).
runpy.run_path(str(Path(__file__).resolve().parents[1] / "src" / "molmo_motion" / "numpy_compat.py"))

MOLMO_MOTION_1M_ROOT = os.environ.get(
    "MOLMO_MOTION_1M_ROOT", "data/molmo-motion-1m")

# Dataset → (split_file, tracks_dir, suffix, is_dict_format)
# is_dict_format = the points_3d array is a 0-d object array wrapping a
#   {obj_name: ndarray} dict. False = flat (T,N,3) or (N,T,3) ndarray.
# Must stay in sync with Trajectory3DDataset._DICT_FORMAT_DATASETS and the
# per-dataset track paths in trajectory_3d_dataset.py. (The internal token
# for HD-EPIC is "hepic"; the on-disk directory is "hdepic".)
DATASETS = {
    "egodex":      ("egodex/annotations/egodex_split.json",            "egodex/tracks/object",    "_3d.npz",  True),
    "hepic":       ("hdepic/annotations/hdepic_split.json",            "hdepic/tracks",           "_3d.npz",  True),
    "ytvis":       ("ytvis/annotations/ytvis_split.json",              "ytvis/tracks",            "_3d.npz",  True),
    "stereo4d":    ("stereo4d/annotations/stereo4d_split.json",        "stereo4d/tracks",         "_3d.npz",  True),
    "molmospaces": ("molmospaces/annotations/molmospaces_split.json",  "molmospaces/tracks",      "_3d.npz",  True),
}


def _scan_one(args):
    """Return (file_id, keys_or_None_if_missing_or_corrupted)."""
    file_id, npz_path = args
    if not os.path.exists(npz_path):
        return file_id, None
    try:
        with np.load(npz_path, allow_pickle=True) as f:
            pts = f["points_3d"]
            if pts.dtype == object:
                keys = sorted(pts.item().keys())
            else:
                keys = ["__flat__"]
    except (EOFError, ValueError, OSError):
        # Corrupted / truncated NPZ — treat as no keys.
        return file_id, None
    return file_id, keys


def build_cache_for_dataset(ds: str, root: Path, cache_path: Path, num_workers: int):
    split_file, tracks_dir, suf, _ = DATASETS[ds]
    splits = json.load(open(root / split_file))
    td = root / tracks_dir

    # Collect unique file_ids across train+test
    seen = set()
    work = []
    for s in ("train", "test"):
        for e in splits.get(s, []):
            fid = e["file"]
            if fid in seen:
                continue
            seen.add(fid)
            work.append((fid, str(td / f"{fid}{suf}")))

    if not work:
        cache_path.write_text("{}")
        return 0, 0

    results = {}
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_scan_one, w): w[0] for w in work}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{ds:12s} scan", mininterval=2.0):
                fid, keys = fut.result()
                results[fid] = keys
    else:
        for w in tqdm(work, desc=f"{ds:12s} scan", mininterval=2.0):
            fid, keys = _scan_one(w)
            results[fid] = keys

    # Persist atomically.
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(results))
    tmp.replace(cache_path)

    n_missing = sum(1 for v in results.values() if v is None)
    n_total = len(results)
    return n_total, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS.keys()),
                    help="Comma-separated dataset names to build")
    ap.add_argument("--cache_dir", type=str,
                    default=os.environ.get(
                        "MOLMO_MOTION_1M_TRACK_KEYS_CACHE",
                        str(Path.home() / ".cache" / "molmo_motion_1m_track_keys")))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if cache exists")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = Path(MOLMO_MOTION_1M_ROOT)

    requested = args.datasets.split(",")
    for ds in requested:
        if ds not in DATASETS:
            print(f"[skip] unknown dataset: {ds}")
            continue
        cp = cache_dir / f"{ds}.json"
        if cp.exists() and not args.force:
            print(f"[skip] {ds}: cache exists at {cp} (use --force to rebuild)")
            continue
        split_file = DATASETS[ds][0]
        if not (root / split_file).exists():
            print(f"[skip] {ds}: split file not found at {root / split_file} "
                  f"— dataset not downloaded under MOLMO_MOTION_1M_ROOT={root}. "
                  f"Pass --datasets to limit to the datasets you downloaded.")
            continue
        print(f"[build] {ds} → {cp}")
        total, missing = build_cache_for_dataset(ds, root, cp, args.workers)
        print(f"  {ds}: {total} entries, {missing} missing/corrupted NPZs")


if __name__ == "__main__":
    main()
