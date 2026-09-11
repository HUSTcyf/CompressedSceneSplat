"""
v3 GLOBAL sign-alignment registry: multi-reference majority vote.

READ-ONLY: never modifies the original npz files. Same output format as v2
(names + signs npz) so the dataset code (global_sign_path) is unchanged.

v3 fixes (code review of v2):
  R1. v2 used a SINGLE reference (00777c41d4_0). Two non-reference chunks could
      disagree (measured: 00a231a370 vs 01ce24e652 pos_frac = 0.44) because
      each is only aligned to the seed, and the seed's class-mean on a dim can
      be noise. v3: majority vote over a reference POOL (chunks with >=5
      classes), weighted by common-class count x the reference's own per-dim
      seed-alignment confidence |corr|.
  R2. v2 let the class-mean correlation sign flip dim0. dim0 is the public
      direction (constant per chunk, max-abs canonicalize anchor is stable);
      its class-mean vector is near-constant over classes -> corr is
      ill-conditioned -> 577/3799 chunks got noise flips on dim0. v3 forces
      dim0 = +1 (the canonicalize convention is already globally consistent).
  R3. v2 skipped chunks with <3 common classes with the SINGLE reference
      (22 chunks: mostly 1-3-class chunks like pure-floor scenes, plus one
      13-class chunk with little overlap). v3: (a) pool of references spreads
      class coverage; (b) K=1-2 fallback via shared-class mean-sign anchor.

Usage:
    /home/isom/.conda/envs/scene_splat/bin/python tools/compression/build_global_sign_registry_v3.py
"""
import sys
import os
import glob
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pointcept.utils.svd_sign import canonicalize_svd_sign  # noqa: E402

ROOTS = [
    "/home/isom/cyf/SceneSplat/scannetpp_v2/train_grid1.0cm_chunk6x6_stride3x3",
    "/home/isom/cyf/SceneSplat/scannetpp_v2/test_grid1.0cm_chunk6x6_stride3x3",
]
RANK = 8
SEED = "00777c41d4_0"
MIN_PTS = 50          # min points per class for a class mean
MIN_COMMON = 3        # min common classes for a correlation sign
MIN_POOL_CLASSES = 5  # reference pool: chunks with >=5 classes
OUT = "/home/isom/cyf/SceneSplat/scannetpp_v2/lang_feat_grid_svd_r8_global_signs_v3.npz"
N_WORKERS = 16


def load_chunk(path):
    d = np.load(os.path.join(path, f"lang_feat_grid_svd_r{RANK}.npz"))
    C = d["compressed"]
    idx = d["indices"]
    seg = np.load(os.path.join(path, "segment.npy"))
    if seg.ndim == 2:
        seg = seg[:, 0]
    seg = seg.astype(np.int32)
    vm = np.load(os.path.join(path, "valid_feat_mask.npy")).astype(bool)
    assert idx.shape[0] == vm.sum(), f"{path}: idx {idx.shape[0]} vs valid {vm.sum()}"
    return C, idx, seg[vm]


def class_means(feat, seg_v):
    means = {}
    for c in np.unique(seg_v):
        m = seg_v == c
        if m.sum() >= MIN_PTS:
            means[int(c)] = feat[m].mean(0)
    return means


def per_dim_corr(mA, mB):
    """Per-dim correlation of class-mean vectors over common classes."""
    common = sorted(set(mA) & set(mB))
    if len(common) < MIN_COMMON:
        return None, None
    A = np.stack([mA[c] for c in common])  # [K,16]
    B = np.stack([mB[c] for c in common])
    corrs = np.zeros(A.shape[1])
    for dd in range(A.shape[1]):
        a, b = A[:, dd], B[:, dd]
        if a.std() < 1e-9 or b.std() < 1e-9:
            corrs[dd] = 0.0
        else:
            corrs[dd] = np.corrcoef(a, b)[0, 1]
    return corrs, len(common)


def pass1_chunk(path):
    """Align to seed (v2 logic). Returns (name, means, s1, c1) or (name, None)."""
    name = os.path.basename(path)
    try:
        C, idx, seg_v = load_chunk(path)
        feat = canonicalize_svd_sign(C)[idx]
        means = class_means(feat, seg_v)
        corrs, k = per_dim_corr(means, SEED_MEANS)
        if corrs is None:
            return (name, means, None, None)  # cannot align to seed; fallback in pass 2
        s1 = np.where(corrs >= 0, 1.0, -1.0)
        s1[corrs == 0] = 1.0
        return (name, means, s1, np.abs(corrs))
    except Exception as e:
        print(f"ERROR {name}: {e}")
        return (name, None, None, None)


def pass2_chunk(args):
    """Majority vote over the reference pool (all in seed convention)."""
    name, means, s1, c1 = args
    if means is None:
        return (name, None)  # unlabeled / unloadable -> not registered
    signs = np.ones(RANK, dtype=np.float32)
    for dd in range(RANK):
        if dd == 0:
            signs[dd] = 1.0  # R2: public direction anchored by canonicalize
            continue
        votes = 0.0
        for rname, rmeans, rs1, rc1 in POOL:
            corrs, k = per_dim_corr(means, rmeans)
            if corrs is None:
                continue
            # R's aligned means: m_R * s1_R (seed convention); vote = sign(corr(B, R_aligned))
            s_align = 1.0 if rs1 is None else rs1[dd]
            v = np.sign(corrs[dd] * s_align) if corrs[dd] != 0 else 0.0
            conf = 1.0 if rc1 is None else rc1[dd]
            votes += k * conf * v
        if votes != 0.0:
            signs[dd] = np.sign(votes)
            continue
        # R3 fallback: K=1-2 shared-class mean-sign anchor
        anchor = 0.0
        for rname, rmeans, rs1, rc1 in POOL:
            common = sorted(set(means) & set(rmeans))
            if not common:
                continue
            s_align = 1.0 if rs1 is None else rs1[dd]
            for c in common:
                v = means[c][dd] * rmeans[c][dd] * s_align
                anchor += v
        if abs(anchor) > 1e-8:
            signs[dd] = np.sign(anchor)
    return (name, signs)


def main():
    global SEED_MEANS, POOL
    dirs = []
    for root in ROOTS:
        dirs += glob.glob(os.path.join(root, "*"))
    dirs = sorted(d for d in dirs
                  if os.path.exists(os.path.join(d, f"lang_feat_grid_svd_r{RANK}.npz")))
    print(f"chunks: {len(dirs)}")

    # Seed means (training space)
    C, idx, seg_v = load_chunk(os.path.join(ROOTS[0], SEED))
    SEED_MEANS = class_means(canonicalize_svd_sign(C)[idx], seg_v)
    print(f"seed {SEED}: classes={len(SEED_MEANS)}")

    # Pass 1: align everything to the seed
    with Pool(N_WORKERS) as pool:
        pass1_items = list(pool.imap_unordered(pass1_chunk, dirs))
    pass1 = {n: (m, s, c) for n, m, s, c in pass1_items}
    n1 = sum(1 for n, v in pass1.items() if v is not None and v[1] is not None)
    print(f"pass1 aligned to seed: {n1}/{len(dirs)}")

    # Reference pool: chunks with >=5 classes AND seed-aligned, sorted by
    # class richness, capped (vote cost is pool-size x 16 dims)
    POOL = []
    for n, v in pass1.items():
        if v is None or isinstance(v, str) or v[0] is None:
            continue
        means, s1, c1 = v
        if len(means) >= MIN_POOL_CLASSES and s1 is not None:
            POOL.append((n, means, s1, c1))
    POOL.sort(key=lambda x: -len(x[1]))
    POOL = POOL[:300]
    print(f"reference pool: {len(POOL)} chunks (>= {MIN_POOL_CLASSES} classes, capped 300)")

    # Pass 2: majority vote
    with Pool(N_WORKERS) as pool:
        pass2 = list(pool.imap_unordered(
            pass2_chunk, [(n, None, None, None) if isinstance(v, str) else (n, *v)
                          for n, v in pass1.items()]))

    names, signs = [], []
    skipped = 0
    for n, s in pass2:
        if s is None:
            skipped += 1
            continue
        names.append(n)
        signs.append(s)
    signs = np.array(signs, dtype=np.float32)
    # Seed's own entry: +1 by definition of the convention
    if SEED in names:
        signs[names.index(SEED)] = 1.0
    np.savez(OUT, names=np.array(names), signs=signs)
    print(f"DONE: registered={len(names)} skipped={skipped} -> {OUT}")

    # Report flip statistics + agreement with v2
    neg = {i: int((signs[:, i] < 0).sum()) for i in range(signs.shape[1])}
    print("per-dim neg counts:", [neg[i] for i in range(signs.shape[1])])
    v2 = os.path.join(os.path.dirname(OUT), "lang_feat_grid_svd_r8_global_signs.npz")
    if os.path.exists(v2):
        d2 = np.load(v2)
        n2 = set(d2["names"].tolist())
        agree = diff = 0
        for i, n in enumerate(names):
            if n in n2:
                j = list(d2["names"]).index(n)
                agree += int((signs[i] == d2["signs"][j]).sum())
                diff += int((signs[i] != d2["signs"][j]).sum())
        print(f"v3 vs v2: agree={agree} diff={diff} ({100*diff/(agree+diff):.1f}% changed)")


if __name__ == "__main__":
    main()
