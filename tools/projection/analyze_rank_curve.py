#!/usr/bin/env python
"""
Analyze energy retention and category separability vs rank from saved Gram matrix.

Loads exp/text_anchor_basis/gram.npy (or singular_values.npy) and plots the
energy retention curve for rank in [1, 768]. Optionally re-encodes scannet200
queries to measure category separability (mean pairwise |cos|) vs rank.

Usage:
    python analyze_rank_curve.py --gram exp/text_anchor_basis/gram.npy \
        --model-dir /home/isom/cyf/models/siglip2-base-patch16-512 \
        --output exp/text_anchor_basis/rank_curve.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gram", default="exp/text_anchor_basis/gram.npy")
    parser.add_argument("--model-dir", default=None, help="if set, also measure separability vs rank")
    parser.add_argument("--output", default="exp/text_anchor_basis/rank_curve.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    G = np.load(args.gram)
    print(f"Gram: {G.shape}")

    evals = np.linalg.eigvalsh(G)[::-1]
    evals = np.clip(evals, 0, None)
    total = evals.sum()
    cumsum = np.cumsum(evals)
    ret_curve = cumsum / total

    report = {"total_energy": float(total)}
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
    print("\n[1] Energy retention curve (anchor corpus, Gram-based):")
    for r in [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768]:
        print(f"  rank {r:4d}: {ret_curve[r-1]:.4f}")
    report["energy_retention_curve"] = {int(r): float(ret_curve[r-1]) for r in [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768]}
    for t in thresholds:
        r = int(np.searchsorted(cumsum, t * total) + 1)
        r = min(r, 768)
        print(f"  threshold {t:.2f} reached at rank {r}")
        report[f"rank_for_{t}"] = r

    # [2] Category separability vs rank (needs encoder)
    if args.model_dir:
        print("\n[2] scannet200 separability vs rank:")
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(args.model_dir, torch_dtype=torch.float16).to(device).eval()
        tok = AutoTokenizer.from_pretrained(args.model_dir)

        labels = []
        with open("/home/isom/cyf/CompressedSceneSplat/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_labels.txt") as f:
            labels = [l.strip() for l in f if l.strip()]
        prompts = [f"this is a {l}" for l in labels]
        with torch.no_grad():
            inputs = tok(prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
            Q = model.get_text_features(**inputs).float()
        Q = Q / Q.norm(dim=-1, keepdim=True)
        Q = Q.cpu().numpy()

        _, V = np.linalg.eigh(G)
        V = V[:, ::-1]
        sim_768 = np.abs(Q @ Q.T)
        np.fill_diagonal(sim_768, 0)
        base = sim_768.mean()
        print(f"  768-dim baseline mean |cos|: {base:.4f}")
        report["baseline_abs_cos_768"] = float(base)
        report["separability_abs_cos"] = {}
        for r in [8, 16, 32, 64, 128, 256, 512, 768]:
            Vr = V[:, :r]
            qr = Q @ Vr
            qr = qr / (np.linalg.norm(qr, axis=-1, keepdims=True) + 1e-12)
            sim = np.abs(qr @ qr.T)
            np.fill_diagonal(sim, 0)
            m = sim.mean()
            print(f"  rank {r:4d}: mean |cos| = {m:.4f}")
            report["separability_abs_cos"][int(r)] = float(m)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
