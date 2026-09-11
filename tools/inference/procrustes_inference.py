"""
Standalone open-vocabulary inference for compressed (16D) LangPretrainer models
with analytic per-scene Procrustes alignment.

Pipeline (all label-free):
1. Model predicts 16D features per point from 3DGS attributes (coord/color/opacity/quat/scale)
2. Text embeddings [M, D_full] reduced with plain SVD (no weighting) -> Y_16 = T @ V_text
3. Scene basis V_i [D_full, 16] from the scene's own lang_feat.npy via plain SVD
4. Analytic Procrustes: Q = V_i^T @ V_text  (orthogonal change of basis, det corrected)
5. Similarity = (X_pred @ Q) @ Y_16^T  -> argmax -> per-point class prediction

Usage:
    python tools/inference/procrustes_inference.py \
        --config configs/inference/lang-pretrain-litept-ovs-gridsvd.py \
        --checkpoint /path/to/model_best.pth \
        --scene-root /path/to/preprocessed/3dgs/npy/folder \
        --text-embeddings /path/to/scannet200_text_embeddings_siglip2.pt \
        --output-dir ./procrustes_output \
        --rank 16 \
        [--svd-basis-path /path/to/scene_basis.npy]   # optional precomputed V_i [768,16]
        [--gt-segment /path/to/segment.npy --class-names /path/to/names.json]  # optional mIoU eval
"""

import argparse
import json
import logging
import os

import numpy as np
import torch

from pointcept.inference import LangPretrainerInference
from pointcept.utils.config import Config


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ProcrustesInference")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def load_text_embeddings(path: str) -> np.ndarray:
    if path.endswith(".pt"):
        data = torch.load(path, weights_only=False)
        if isinstance(data, dict):
            if "embeddings" in data:
                emb = data["embeddings"]
            elif "text_embeddings" in data:
                emb = data["text_embeddings"]
            else:
                raise KeyError(f"Unknown key in {path}: {list(data.keys())}")
        else:
            emb = data
        return np.asarray(emb.detach().cpu()).astype(np.float32)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    raise ValueError(f"Unsupported text embeddings format: {path}")


def svd_basis(X: np.ndarray, rank: int) -> np.ndarray:
    """Top-`rank` right singular vectors of X via Gram matrix (plain SVD, no weighting).

    X: [N, D] -> returns V [D, rank] with orthonormal columns.
    """
    G = X.T @ X  # [D, D]
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(eigvals)[::-1][:rank]
    return np.ascontiguousarray(eigvecs[:, idx], dtype=np.float32)


def analytic_procrustes(V_scene: np.ndarray, V_text: np.ndarray) -> np.ndarray:
    """Q = V_scene^T @ V_text, det-corrected to a proper rotation (det = +1)."""
    Q = V_scene.T @ V_text  # [16, 16]
    U, S, Vt = np.linalg.svd(Q, full_matrices=False)
    if np.linalg.det(U @ Vt) < 0:
        Vt[-1] *= -1.0
    Q_orth = (U @ Vt).astype(np.float32)
    return Q_orth


def normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Inference config (must match training).")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint (.pth) to load.")
    parser.add_argument("--scene-root", required=True, help="Preprocessed 3DGS .npy scene folder.")
    parser.add_argument("--text-embeddings", required=True, help="Text embeddings (.pt or .npy), [M, D].")
    parser.add_argument("--output-dir", required=True, help="Directory for predictions/report.")
    parser.add_argument("--rank", type=int, default=16, help="SVD rank (must match training).")
    parser.add_argument("--svd-basis-path", default=None,
                        help="Optional precomputed scene basis V_i [D, rank] (.npy). "
                             "If not given, computed from the scene's lang_feat.npy.")
    parser.add_argument("--gt-segment", default=None,
                        help="Optional GT labels for mIoU report, aligned to lang_feat rows.")
    parser.add_argument("--class-names", default=None,
                        help="Optional JSON list of class names for mIoU report.")
    parser.add_argument("--save-sim", action="store_true", help="Save full [N, M] similarity matrix.")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda:0).")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    cfg = Config.fromfile(args.config)
    if not os.path.isdir(args.scene_root):
        raise FileNotFoundError(f"Scene root not found: {args.scene_root}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Model forward: predict 16D features for all points
    # ---------------------------------------------------------------
    inferencer = LangPretrainerInference(cfg, args.checkpoint, device=args.device)
    scene_name = os.path.basename(os.path.normpath(args.scene_root))

    data_dict = {}
    for file_name in sorted(os.listdir(args.scene_root)):
        if not file_name.endswith(".npy"):
            continue
        key = os.path.splitext(file_name)[0]
        if key.startswith("lang_feat"):
            continue  # not a model input
        data_dict[key] = np.load(os.path.join(args.scene_root, file_name))

    missing = [k for k in cfg.get("feat_keys", ()) if k not in data_dict]
    if missing:
        raise FileNotFoundError(f"Missing model input files: {missing}")

    outputs = inferencer(data_dict, scene_name=scene_name, save=False)
    X_pred = outputs["backbone_features"].astype(np.float32)  # [N, 16], already L2-normalized
    logger.info(f"Model predictions: {X_pred.shape}")

    # ---------------------------------------------------------------
    # 2. Text embeddings: plain SVD -> V_text, Y_16 = T @ V_text
    # ---------------------------------------------------------------
    T_full = load_text_embeddings(args.text_embeddings)
    T_norm = normalize_rows(T_full)
    V_text = svd_basis(T_norm, args.rank)  # [D, rank]
    Y_16 = T_norm @ V_text  # [M, rank]
    Y_16 = normalize_rows(Y_16)
    logger.info(f"Text embeddings: {T_full.shape} -> Y_16 {Y_16.shape}")

    # ---------------------------------------------------------------
    # 3. Scene basis V_i (plain SVD of the scene's own lang_feat)
    # ---------------------------------------------------------------
    if args.svd_basis_path is not None:
        V_scene = np.load(args.svd_basis_path).astype(np.float32)
        if V_scene.shape[1] != args.rank:
            raise ValueError(f"Basis rank {V_scene.shape[1]} != {args.rank}")
        logger.info(f"Loaded scene basis: {V_scene.shape}")
    else:
        feat_path = os.path.join(args.scene_root, "lang_feat.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                "No lang_feat.npy in scene; provide --svd-basis-path instead."
            )
        lang_feat = np.load(feat_path).astype(np.float32)
        n_coord = data_dict["coord"].shape[0]
        if lang_feat.shape[0] != n_coord and "valid_feat_mask" in data_dict:
            valid = data_dict["valid_feat_mask"].astype(bool)
            if lang_feat.shape[0] == int(valid.sum()):
                lang_feat_full = np.zeros((n_coord, lang_feat.shape[1]), dtype=np.float32)
                lang_feat_full[valid] = lang_feat
                lang_feat = lang_feat_full
        logger.info(f"Scene lang_feat: {lang_feat.shape}")
        V_scene = svd_basis(lang_feat, args.rank)  # [D, rank]
        logger.info(f"Computed scene basis: {V_scene.shape}")

    # ---------------------------------------------------------------
    # 4. Analytic Procrustes Q and 5. similarity
    # ---------------------------------------------------------------
    Q = analytic_procrustes(V_scene, V_text)  # [16, 16]
    logger.info(f"Q: {Q.shape}, det(Q)={np.linalg.det(Q):.6f}, "
                f"orth_err={np.linalg.norm(Q.T @ Q - np.eye(args.rank)):.2e}")

    X_aligned = X_pred @ Q  # [N, rank]
    sim = X_aligned @ Y_16.T  # [N, M]
    pred = sim.argmax(axis=1).astype(np.int64)

    np.save(os.path.join(args.output_dir, f"{scene_name}_pred.npy"), pred)
    if args.save_sim:
        np.save(os.path.join(args.output_dir, f"{scene_name}_sim.npy"), sim)

    report = {
        "scene": scene_name,
        "num_points": int(X_pred.shape[0]),
        "num_classes": int(Y_16.shape[0]),
        "Q_det": float(np.linalg.det(Q)),
        "Q_orth_error": float(np.linalg.norm(Q.T @ Q - np.eye(args.rank))),
        "pred_file": f"{scene_name}_pred.npy",
    }

    # ---------------------------------------------------------------
    # Optional: mIoU vs GT segment
    # ---------------------------------------------------------------
    if args.gt_segment is not None:
        gt = np.load(args.gt_segment)
        if gt.ndim == 2:
            gt = gt[:, 0]
        gt = gt.reshape(-1).astype(np.int64)
        n_pred = data_dict["coord"].shape[0]
        if gt.shape[0] != n_pred and "valid_feat_mask" in data_dict:
            valid = data_dict["valid_feat_mask"].astype(bool)
            gt_full = np.full(n_pred, -1, dtype=np.int64)
            gt_full[valid] = gt
            gt = gt_full
        if gt.shape[0] != pred.shape[0]:
            raise ValueError(f"GT {gt.shape[0]} != predictions {pred.shape[0]}")

        class_names = None
        if args.class_names and os.path.exists(args.class_names):
            with open(args.class_names) as f:
                class_names = json.load(f)

        valid_gt = gt >= 0
        pred_valid = pred[valid_gt]
        gt_valid = gt[valid_gt]
        ious = []
        per_class = {}
        for c in np.unique(gt_valid):
            inter = np.sum((pred_valid == c) & (gt_valid == c))
            union = np.sum((pred_valid == c) | (gt_valid == c))
            iou = inter / (union + 1e-9)
            ious.append(iou)
            name = class_names[c] if class_names and c < len(class_names) else str(c)
            per_class[name] = float(iou)
        report["mIoU"] = float(np.mean(ious)) if ious else None
        report["per_class_IoU"] = per_class
        report["num_gt_points"] = int(valid_gt.sum())
        logger.info(f"mIoU over {len(ious)} GT classes: {report['mIoU']:.4f}")

    with open(os.path.join(args.output_dir, f"{scene_name}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Done. Report: {os.path.join(args.output_dir, f'{scene_name}_report.json')}")


if __name__ == "__main__":
    main()
