"""
Estimate per-point normals for a 3DGS chunk via KNN PCA (numpy/scipy only).

Follows the same spirit as the original SceneSplat preprocessing
(preprocess_scannet_gs.py: mesh vertex normals assigned to gaussians by KNN;
preprocess_scannetpp_gs.py: point-cloud normals mapped to gaussians).
Here the gaussian point cloud itself is used: the normal of a point = the
smallest-eigenvalue eigenvector of its K-nearest-neighbor covariance.

Sign convention (no camera): per-point max-abs canonicalization — the
dominant component is made positive (deterministic, same idea as
canonicalize_svd_sign). A global flip is harmless for an input feature
(the model sees the same convention in train and eval for a given chunk).

Usage:
    /home/isom/.conda/envs/scene_splat/bin/python tools/compression/estimate_normals.py <chunk_dir> [k]
"""
import numpy as np
import os
import sys
from scipy.spatial import cKDTree


def estimate_normals(coord, k=30):
    """coord: [N,3] -> normals [N,3] (unit length, max-abs positive)."""
    tree = cKDTree(coord)
    d, idx = tree.query(coord, k=min(k, len(coord)))
    nbr = coord[idx]  # [N, k, 3]
    nbr = nbr - nbr.mean(axis=1, keepdims=True)
    # covariance [N, 3, 3]
    cov = np.einsum("nki,nkj->nij", nbr, nbr) / nbr.shape[1]
    # smallest eigenvector via eigh (ascending eigenvalues)
    w, v = np.linalg.eigh(cov)  # v: [N, 3, 3], columns = eigenvectors
    n = v[:, :, 0]  # smallest eigenvalue's eigenvector
    # normalize
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    # sign: max-abs component positive (deterministic convention)
    imax = np.argmax(np.abs(n), axis=1)
    signs = np.take_along_axis(n, imax[:, None], axis=1)[:, 0]
    n = n * np.sign(signs)[:, None]
    return n.astype(np.float32)


def main():
    chunk = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    coord = np.load(os.path.join(chunk, "coord.npy")).astype(np.float32)
    print(f"points: {len(coord)}, k={k}")
    n = estimate_normals(coord, k)
    out = os.path.join(chunk, "normal.npy")
    np.save(out, n)
    print(f"saved {out}: {n.shape} {n.dtype}")
    # sanity: unit norm, local consistency (KNN dot product)
    tree = cKDTree(coord)
    _, idx = tree.query(coord[:200000], k=6)
    dots = np.abs((n[:200000, None, :] * n[idx[:, 1:]]).sum(axis=2)).mean()
    print(f"KNN normal agreement (|dot| mean): {dots:.3f} (1.0 = perfectly consistent)")


if __name__ == "__main__":
    main()
