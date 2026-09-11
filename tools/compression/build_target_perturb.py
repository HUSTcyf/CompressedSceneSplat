"""
Build training-target perturbation sidecars for rebuttal Table 2 (2026-08-04).

Mode "flip":   fixed-seed random +/-1 vector, SAME for all scenes (a single
               equivalent basis). Output format identical to the global_signs
               registry (names + signs), so it plugs into the existing
               `global_sign_path` config key — zero dataset code changes.
Mode "rotate": fixed random orthogonal matrix (scipy.stats.ortho_group.rvs,
               seed 0), saved as npz key "matrix" -> `target_rotate_matrix_path`
               config key (new kwarg in ScanNetPPGSDataset).

Both perturbations are applied AFTER canonicalize in the dataset loader, so they
are genuine equivalent-basis changes of the training target (eval side is
unaffected: val/test load_compressed_lang_feat=False, and the Procrustes Q fit
absorbs linear transforms of the prediction).
"""
import argparse
import os

import numpy as np

REG_V3 = "/home/isom/cyf/SceneSplat/scannetpp_v2/lang_feat_grid_svd_r16_global_signs_v3.npz"
OUT_DIR = "/home/isom/cyf/SceneSplat/scannetpp_v2"
RANK = 16
SEED = 0


def build_flip(out):
    reg = np.load(REG_V3)
    names = reg["names"]  # covers all train+test grid chunks (same split as training)
    rng = np.random.RandomState(SEED)
    vec = rng.choice([-1.0, 1.0], size=RANK).astype(np.float32)
    signs = np.repeat(vec[None, :], len(names), axis=0)
    np.savez(out, names=names, signs=signs)
    n_neg = int((vec < 0).sum())
    print(f"flip: {len(names)} chunks, same vector for all, neg-dims={n_neg}/{RANK}, "
          f"vector={vec.tolist()} -> {out}")


def build_rotate(out):
    from scipy.stats import ortho_group

    m = ortho_group.rvs(RANK, random_state=SEED).astype(np.float32)
    err = np.abs(m @ m.T - np.eye(RANK)).max()
    assert err < 1e-4, f"not orthogonal: {err}"
    np.savez(out, matrix=m)
    print(f"rotate: {m.shape} orth-err={err:.2e} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["flip", "rotate"])
    args = ap.parse_args()
    if args.mode == "flip":
        build_flip(os.path.join(OUT_DIR, f"lang_feat_grid_svd_r{RANK}_flip_signs.npz"))
    else:
        build_rotate(os.path.join(OUT_DIR, f"lang_feat_grid_svd_r{RANK}_rotate_matrix.npz"))
