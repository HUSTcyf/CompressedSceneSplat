"""
Select top-50 train chunks whose class distribution is closest to the val
split (2026-08-04 training strategy change: 50 chunks x 100 iterations each).

Similarity: cosine between normalized class histograms (segment.npy, top100
benchmark encoding). Chunks are 6x6m grid chunks; val is full scenes.

Outputs:
  /home/isom/cyf/SceneSplat/scannetpp_v2/top50_chunks.txt  (relpaths, one per line)
  prints: top-20 scores, scene coverage, class overlap with val
"""
import glob
import os
from multiprocessing import Pool

import numpy as np

ROOT = "/home/isom/cyf/SceneSplat/scannetpp_v2"
TRAIN_ROOTS = [
    os.path.join(ROOT, "train_grid1.0cm_chunk6x6_stride3x3"),
    os.path.join(ROOT, "test_grid1.0cm_chunk6x6_stride3x3"),
]
VAL_ROOT = os.path.join(ROOT, "val")
N_CLASSES = 100
TOP_K = 50
N_WORKERS = 12


def load_seg(path):
    seg = np.load(path)
    if seg.ndim == 2:
        seg = seg[:, 0]
    return seg.astype(np.int64)


def hist(seg, n=N_CLASSES):
    seg = seg[seg >= 0]  # ignore labels (-1) excluded
    h = np.bincount(seg, minlength=n)[:n].astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else h


def val_histogram():
    vh = np.zeros(N_CLASSES)
    n_scenes = 0
    for d in sorted(glob.glob(os.path.join(VAL_ROOT, "*"))):
        if not os.path.isdir(d):
            continue
        p = os.path.join(d, "segment.npy")
        if os.path.exists(p):
            vh += hist(load_seg(p))
            n_scenes += 1
    vh /= vh.sum()
    print(f"val: {n_scenes} scenes, classes present: {(vh > 0).sum()}")
    return vh


def score_chunk(args):
    d, vh = args
    p = os.path.join(d, "segment.npy")
    if not os.path.exists(p):  # some test_grid chunks have no labels
        return None
    h = hist(load_seg(p))
    denom = np.linalg.norm(h) * np.linalg.norm(vh)
    cos = float(h @ vh / denom) if denom > 0 else 0.0
    return cos, d


def main():
    vh = val_histogram()
    dirs = []
    for root in TRAIN_ROOTS:
        dirs += [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    print(f"chunks: {len(dirs)}")

    with Pool(N_WORKERS) as pool:
        scores = list(pool.imap_unordered(score_chunk, [(d, vh) for d in dirs], chunksize=32))
    scores = [s for s in scores if s is not None]
    print(f"chunks with labels: {len(scores)}/{len(dirs)}")
    scores.sort(key=lambda x: -x[0])
    top = scores[:TOP_K]

    print("=== top-20 ===")
    for cos, d in top[:20]:
        print(f"  {cos:.4f}  {os.path.relpath(d, ROOT)}")

    names = [os.path.relpath(d, ROOT) for _, d in top]
    scenes = sorted({os.path.basename(d).split("_")[0] for _, d in top})
    print(f"\ntop-{TOP_K} similarity range: {top[-1][0]:.4f} .. {top[0][0]:.4f}")
    print(f"scenes covered: {len(scenes)}/{len(scenes)} unique: {len(scenes)}")
    print(f"scene list: {scenes}")

    # merged class coverage of selected chunks vs val
    sel = np.zeros(N_CLASSES)
    for _, d in top:
        sel += hist(load_seg(os.path.join(d, "segment.npy")))
    sel /= sel.sum()
    cos_sel = float(sel @ vh / (np.linalg.norm(sel) * np.linalg.norm(vh)))
    print(f"merged selected-chunk hist vs val: cos={cos_sel:.4f}, "
          f"val classes: {(vh > 0).sum()}, selected classes: {(sel > 0).sum()}")

    out = os.path.join(ROOT, "top50_chunks.txt")
    with open(out, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
