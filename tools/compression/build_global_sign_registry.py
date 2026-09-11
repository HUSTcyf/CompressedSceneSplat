"""
Build a GLOBAL sign-alignment registry for per-chunk SVD-compressed features.

READ-ONLY: never modifies the original npz files. Computes, for every chunk,
the per-dim sign (+-1) that aligns its columns to ONE reference chunk
(00777c41d4_0), using the sign of the correlation of class-mean vectors over
common classes. Output is a single sidecar npz:

    lang_feat_grid_svd_r16_global_signs.npz
        names: [K] chunk dir basenames (str array)
        signs: [K, 16] float32 sign per chunk per dim

The dataset (scannetppgs.py) applies `compressed * signs[name]` at LOAD time
when global_sign_path is configured — original data untouched, fully
reversible (remove the config flag -> original behavior).

Why: per-chunk SVD bases have arbitrary column signs. The compression-time
canonicalize (max-abs anchor) only fixes strong dims; weak dims keep random
signs per chunk -> multi-chunk training gradients cancel -> model learns ~0
waves (per-dim corr ~0) -> L1 plateaus at full wave magnitude ("the
bottleneck"). Validated by diag_wave_basis_consistency.py: after global
alignment, ALL 16 dims reach frac_pos=1.00 across scenes.

Usage:
    /home/isom/.conda/envs/scene_splat/bin/python tools/compression/build_global_sign_registry.py
"""
import numpy as np
import os
import glob
import sys
from multiprocessing import Pool

# 确保能 import pointcept（脚本可能从任意 cwd 运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOTS = [
    "/home/isom/cyf/SceneSplat/scannetpp_v2/train_grid1.0cm_chunk6x6_stride3x3",
    "/home/isom/cyf/SceneSplat/scannetpp_v2/test_grid1.0cm_chunk6x6_stride3x3",
]
RANK = 16
REF_CHUNK = "00777c41d4_0"
MIN_PTS = 50
MIN_COMMON = 3
N_WORKERS = 16
OUT = "/home/isom/cyf/SceneSplat/scannetpp_v2/lang_feat_grid_svd_r16_global_signs.npz"

ref_dir = os.path.join(ROOTS[0], REF_CHUNK)


def load_chunk(path):
    d = np.load(os.path.join(path, f"lang_feat_grid_svd_r{RANK}.npz"))
    C = d["compressed"]
    idx = d["indices"]
    seg = np.load(os.path.join(path, "segment.npy"))
    if seg.ndim == 2:
        seg = seg[:, 0]
    seg = seg.astype(np.int32)
    vm = np.load(os.path.join(path, "valid_feat_mask.npy")).astype(bool)
    return C, idx, seg[vm]


def canonicalize(C):
    """Same canonicalization the dataset applies at load time (svd_sign.py),
    so the registry lives in the TRAINING space, not the raw npz space."""
    from pointcept.utils.svd_sign import canonicalize_svd_sign
    return canonicalize_svd_sign(C)


def class_means(feat, seg_v):
    means = {}
    for c in np.unique(seg_v):
        m = seg_v == c
        if m.sum() >= MIN_PTS:
            means[int(c)] = feat[m].mean(0)
    return means


def compute_ref_means():
    C, idx, seg_v = load_chunk(ref_dir)
    return class_means(canonicalize(C)[idx], seg_v)


def process_chunk(path):
    """Return (basename, signs) or (basename, None) if not alignable."""
    name = os.path.basename(path)
    try:
        C, idx, seg_v = load_chunk(path)
        means = class_means(canonicalize(C)[idx], seg_v)  # training space
        common = sorted(set(means) & set(REF_MEANS))
        if len(common) < MIN_COMMON:
            return (name, None)
        A = np.stack([means[c] for c in common])       # [K,16]
        R = np.stack([REF_MEANS[c] for c in common])   # [K,16]
        signs = np.ones(16, dtype=np.float32)
        for d in range(16):
            a, r = A[:, d], R[:, d]
            s = np.sign(np.dot(a - a.mean(), r - r.mean()))
            if s != 0:
                signs[d] = s
        return (name, signs)
    except Exception as e:
        return (name, f"ERROR: {e}")


def main():
    global REF_MEANS
    REF_MEANS = compute_ref_means()
    print(f"reference: {REF_CHUNK}, classes: {len(REF_MEANS)}")
    dirs = []
    for root in ROOTS:
        dirs += glob.glob(os.path.join(root, "*"))
    dirs = sorted(d for d in dirs
                  if os.path.exists(os.path.join(d, f"lang_feat_grid_svd_r{RANK}.npz")))
    print(f"chunks to process: {len(dirs)}")

    names, signs = [], []
    applied = skipped = 0
    errors = []
    with Pool(N_WORKERS) as pool:
        for i, (name, s) in enumerate(pool.imap_unordered(process_chunk, dirs)):
            if s is None:
                skipped += 1
            elif isinstance(s, str):
                errors.append((name, s))
            else:
                names.append(name)
                signs.append(s)
                applied += 1
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(dirs)} applied={applied} skipped={skipped}")

    signs = np.array(signs, dtype=np.float32)
    np.savez(OUT, names=np.array(names), signs=signs)
    print(f"DONE: applied={applied} skipped={skipped} errors={len(errors)}")
    print(f"saved: {OUT}")
    for n, e in errors[:10]:
        print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
