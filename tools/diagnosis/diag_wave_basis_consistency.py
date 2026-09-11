"""
Diag: per-dim class-mean consistency of SVD-compressed wave features across chunks.

Question: in multi-chunk training, the L1 gradient on each dim averages over chunks.
If dim d's class-mean direction is inconsistent across chunks (per-scene SVD bases
rotated/permuted), the gradient cancels -> model outputs ~0 waves -> corr ~0 -> L1
plateaus at the full wave magnitude ("the bottleneck").

This script measures exactly what the training sees: for each dim, the correlation
of class-mean vectors between chunk pairs:
  - same-scene pairs (00777c41d4_i vs 00777c41d4_j): bases should be near-consistent
  - cross-scene pairs (00777c41d4_0 vs 00a231a370_0): per-scene bases -> inconsistent?

If cross-scene per-dim consistency is ~0 while same-scene is high, the multi-chunk
L1/cos bottleneck is explained by basis inconsistency (NOT a training-code bug),
and the single-chunk overfit learning (corr 0.2-0.6) is consistent with code being
able to learn waves when the basis is consistent.

Usage:
    /home/isom/.conda/envs/scene_splat/bin/python tools/diagnosis/diag_wave_basis_consistency.py
"""
import numpy as np
import os
import glob

ROOT = "/home/isom/cyf/SceneSplat/scannetpp_v2/train_grid1.0cm_chunk6x6_stride3x3"
RANK = 16
MIN_PTS = 50


def load_chunk(path):
    d = np.load(os.path.join(path, f"lang_feat_grid_svd_r{RANK}.npz"))
    C = d["compressed"]  # [M, 16] grid-cell features (canonicalized per chunk)
    idx = d["indices"]   # [V] valid-point -> cell idx (len == num valid points)
    seg = np.load(os.path.join(path, "segment.npy"))
    if seg.ndim == 2:
        seg = seg[:, 0]
    seg = seg.astype(np.int32)  # [N] all points
    vm = np.load(os.path.join(path, "valid_feat_mask.npy")).astype(bool)  # [N]
    assert idx.shape[0] == vm.sum(), f"{idx.shape[0]} vs {vm.sum()}"
    feat = C[idx]  # [V, 16] per-point features, same as training target
    seg_v = seg[vm]  # [V]
    return feat, seg_v


def class_means(feat, seg_v):
    means = {}
    for c in np.unique(seg_v):
        m = seg_v == c
        if m.sum() >= MIN_PTS:
            means[int(c)] = feat[m].mean(0)
    return means


def per_dim_corr(mA, mB):
    common = sorted(set(mA) & set(mB))
    if len(common) < 3:
        return None, len(common)
    A = np.stack([mA[c] for c in common])  # [K, 16]
    B = np.stack([mB[c] for c in common])
    corrs = np.full(16, np.nan)
    for d in range(16):
        a, b = A[:, d], B[:, d]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        corrs[d] = np.corrcoef(a, b)[0, 1]
    return corrs, len(common)


def sign_fix(feat, seg_v):
    """Simulate re-canonicalization: per dim, anchor sign on the most frequent
    class's class-mean (deployable per-chunk, no cross-chunk info)."""
    out = feat.copy()
    means = class_means(feat, seg_v)
    if not means:
        return out
    # most frequent class = class with most points
    counts = {c: (seg_v == c).sum() for c in means}
    ref = max(counts, key=counts.get)
    m_ref = means[ref]  # [16]
    for d in range(16):
        if m_ref[d] != 0:
            out[:, d] = out[:, d] * np.sign(m_ref[d])
    return out


def main():
    all_dirs = sorted(glob.glob(os.path.join(ROOT, "*")))
    # same-scene chunks
    ss = [d for d in all_dirs if os.path.basename(d).startswith("00777c41d4")]
    # cross-scene chunks (different scene, same chunk index 0)
    other = [d for d in all_dirs if not os.path.basename(d).startswith("00777c41d4")]
    cs = [d for d in other if os.path.basename(d).endswith("_0")][:3]

    print(f"same-scene chunks: {[os.path.basename(d) for d in ss]}")
    print(f"cross-scene chunks: {[os.path.basename(d) for d in cs]}")

    cache = {}
    def get(d):
        if d not in cache:
            cache[d] = load_chunk(d)
        return cache[d]

    # Same-scene pairs (within 00777c41d4)
    print("\n=== SAME-SCENE pairs (00777c41d4_i vs 00777c41d4_j) ===")
    ss_corrs = []
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            fi, si = get(ss[i]); fj, sj = get(ss[j])
            mA, mB = class_means(fi, si), class_means(fj, sj)
            c, k = per_dim_corr(mA, mB)
            if c is None:
                continue
            ss_corrs.append(c)
            print(f"  {os.path.basename(ss[i])} vs {os.path.basename(ss[j])}: "
                  f"K={k:3d} mean|corr|={np.nanmean(np.abs(c)):.3f} "
                  f"mean corr={np.nanmean(c):.3f} dims1-15 mean|corr|={np.nanmean(np.abs(c[1:])):.3f}")

    # Cross-scene pairs
    print("\n=== CROSS-SCENE pairs ===")
    cs_corrs = []
    for di in [ss[0]]:
        for dj in cs:
            fi, si = get(di); fj, sj = get(dj)
            mA, mB = class_means(fi, si), class_means(fj, sj)
            c, k = per_dim_corr(mA, mB)
            if c is None:
                continue
            cs_corrs.append(c)
            print(f"  {os.path.basename(di)} vs {os.path.basename(dj)}: "
                  f"K={k:3d} mean|corr|={np.nanmean(np.abs(c)):.3f} "
                  f"mean corr={np.nanmean(c):.3f} dims1-15 mean|corr|={np.nanmean(np.abs(c[1:])):.3f}")

    # Per-dim breakdown for the strongest same/cross pairs
    if ss_corrs:
        ss_all = np.array(ss_corrs)
        print("\nper-dim |corr| same-scene (mean over pairs):")
        print("  " + " ".join(f"d{i}:{np.nanmean(np.abs(ss_all[:, i])):.2f}" for i in range(16)))
    if cs_corrs:
        cs_all = np.array(cs_corrs)
        print("per-dim |corr| cross-scene (mean over pairs):")
        print("  " + " ".join(f"d{i}:{np.nanmean(np.abs(cs_all[:, i])):.2f}" for i in range(16)))

    # =====================================================================
    # Mechanism test: SIGNED per-dim consistency. |corr| high but signed corr
    # ~0 means per-chunk random column signs -> training gradient cancels.
    # =====================================================================
    def signed_summary(pair_corrs, label):
        if not pair_corrs:
            return
        P = np.array(pair_corrs)  # [pairs, 16]
        print(f"\n[{label}] per-dim SIGNED corr (mean over pairs) / frac_positive:")
        for i in range(16):
            col = P[:, i]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                continue
            frac = (col > 0).mean()
            print(f"  d{i}: mean={np.nanmean(col):+.3f} frac_pos={frac:.2f}")

    signed_summary(ss_corrs, "same-scene RAW")
    signed_summary(cs_corrs, "cross-scene RAW")

    # =====================================================================
    # Simulated fix: GLOBAL sign alignment. Every chunk's per-dim sign is
    # aligned to ONE reference chunk (00777c41d4_0) using the sign of the
    # correlation of class-mean vectors over common classes. This is
    # deployable per-chunk WITHOUT raw 768-dim features. If signed corr
    # jumps to 0.5-0.9 on all dims, the sign ambiguity is the bottleneck
    # and the fix is validated.
    # =====================================================================
    print("\n=== GLOBAL sign alignment simulation (ref = 00777c41d4_0) ===")
    ref_feat, ref_seg = get(ss[0])
    ref_means = class_means(ref_feat, ref_seg)

    def global_align(feat, seg_v):
        means = class_means(feat, seg_v)
        common = sorted(set(means) & set(ref_means))
        if len(common) < 3:
            return feat
        A = np.stack([means[c] for c in common])      # [K,16]
        R = np.stack([ref_means[c] for c in common])  # [K,16]
        out = feat.copy()
        for d in range(16):
            a, r = A[:, d], R[:, d]
            s = np.sign(np.dot(a - a.mean(), r - r.mean()))
            if s != 0:
                out[:, d] = out[:, d] * s
        return out

    ga_cache = {}
    def get_ga(d):
        if d not in ga_cache:
            feat, segv = get(d)
            ga_cache[d] = (global_align(feat, segv), segv)
        return ga_cache[d]

    ss_ga, cs_ga = [], []
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            fi, si = get_ga(ss[i]); fj, sj = get_ga(ss[j])
            c, k = per_dim_corr(class_means(fi, si), class_means(fj, sj))
            if c is not None:
                ss_ga.append(c)
    for dj in cs:
        fi, si = get_ga(ss[0]); fj, sj = get_ga(dj)
        c, k = per_dim_corr(class_means(fi, si), class_means(fj, sj))
        if c is not None:
            cs_ga.append(c)

    signed_summary(ss_ga, "same-scene GLOBAL-ALIGNED")
    signed_summary(cs_ga, "cross-scene GLOBAL-ALIGNED")


if __name__ == "__main__":
    main()
