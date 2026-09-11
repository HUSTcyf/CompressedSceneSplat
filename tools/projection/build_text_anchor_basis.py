#!/usr/bin/env python
"""
Build a global text-anchor SVD basis V_T from SigLIP's text training distribution.

Method: exact Gram-matrix accumulation over minibatches.
    X = encoded anchor corpus (rows L2-normalized, [N, 768])
    G = X^T X  (accumulated batch-wise, exact)
    V_T, sigma^2 = eigh(G)  (top-rank eigenvectors / eigenvalues)

Memory: O(d^2) = 768x768 fp32 (~2.3 MB) regardless of N. One pass, exact.

Validation reported:
    1. Energy retention of the anchor corpus itself (sum top-r sigma^2 / total)
    2. Energy retention of evaluation queries (LERF / 3DOVS / ScanNet-200),
       using the same prompt convention as at inference ("this is a {label}")
    3. Category separability: mean pairwise cosine in 16-dim vs 768-dim

Outputs (saved under --output):
    V_T.npy         [768, rank] orthogonal text-anchor basis
    report.json     validation numbers
    gram.npy        [768, 768] accumulated Gram matrix (for reproducibility)
    singular_values.npy [768] full spectrum

Usage (on server, no network needed):
    python tools/projection/build_text_anchor_basis.py \
        --text-file /home/isom/cyf/SceneSplat/laion_corpus.txt \
        --model-dir /home/isom/cyf/models/siglip2-base-patch16-512 \
        --rank 16 --batch-size 100000 --max-texts 4000000 \
        --output exp/text_anchor_basis
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

SCANNET200_LABELS = "/home/isom/cyf/CompressedSceneSplat/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_labels.txt"
SCANNET20_LABELS = "/home/isom/cyf/CompressedSceneSplat/pointcept/datasets/preprocessing/scannet/meta_data/scannet20_labels.txt"


def get_lerf_ovs_labels():
    return [
        "apple", "bag", "bag of cookies", "bear nose", "bowl", "cabinet", "chopsticks",
        "coffee", "coffee mug", "corn", "dall-e brand", "dark cup", "egg", "frog cup",
        "glass of water", "green apple", "green toy chair", "hand", "hooves", "jake",
        "kamaboko", "ketchup", "knife", "miffy", "napkin", "nori", "old camera",
        "onion segments", "ottolenghi", "paper napkin", "pikachu", "pink ice cream",
        "pirate hat", "plastic ladle", "plate", "porcelain hand", "pot", "pour-over vessel",
        "pumpkin", "red apple", "red cup", "red toy chair", "refrigerator", "rubber duck with buoy",
        "rubber duck with hat", "rubics cube", "sake cup", "sheep", "sink", "spatula", "spoon",
        "stuffed bear", "tea in a glass", "tesla door handle", "three cookies", "toaster",
        "toy cat statue", "toy elephant", "waldo", "wavy noodles", "yellow desk", "yellow pouf",
    ]


def get_3dovs_labels():
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ]


def load_labels(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def text_batches(path, batch_size, max_texts=None):
    """Yield batches of texts (one per line), stopping at max_texts."""
    buf = []
    total = 0
    with open(path) as f:
        for line in f:
            t = line.strip()
            if len(t) < 5:
                continue
            buf.append(t)
            if len(buf) == batch_size:
                yield buf
                total += len(buf)
                buf = []
                if max_texts and total >= max_texts:
                    return
    if buf:
        yield buf


@torch.no_grad()
def encode_texts(texts, device, batch_size=512, add_prefix=False):
    prompts = [f"this is a {t}" for t in texts] if add_prefix else texts
    embs = []
    for i in range(0, len(prompts), batch_size):
        inputs = TOKENIZER(
            prompts[i : i + batch_size],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out = MODEL.get_text_features(**inputs).float()
        out = out / out.norm(dim=-1, keepdim=True)
        embs.append(out)
    return torch.cat(embs, dim=0)


def energy_retention(Q, V_T):
    proj = Q @ V_T @ V_T.T
    return (proj * proj).sum(dim=-1).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True, help="Anchor corpus (one alt-text per line)")
    parser.add_argument("--model-dir", required=True, help="Local siglip2-base-patch16-512 dir")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=100000, help="Anchor texts per Gram-accumulation batch")
    parser.add_argument("--encode-batch", type=int, default=512, help="Encoder forward batch")
    parser.add_argument("--max-texts", type=int, default=4000000)
    parser.add_argument("--output", default="exp/text_anchor_basis")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    global MODEL, TOKENIZER
    from transformers import AutoModel, AutoTokenizer
    MODEL = AutoModel.from_pretrained(args.model_dir, torch_dtype=torch.float16).to(device).eval()
    TOKENIZER = AutoTokenizer.from_pretrained(args.model_dir)

    print("=" * 70)
    print("TEXT-ANCHOR SVD BASIS (Gram accumulation, exact)")
    print("=" * 70)
    print(f"anchor corpus : {args.text_file}")
    print(f"model dir     : {args.model_dir}  (device={device})")
    print(f"rank          : {args.rank} | gram-batch {args.batch_size} | max {args.max_texts}")

    print("\nLoading SigLIP2 text encoder...")

    # ---- 1. Gram accumulation (exact, single pass) ----
    d = MODEL.config.text_config.hidden_size
    print(f"text hidden dim: {d}")
    G = torch.zeros(d, d, dtype=torch.float32, device=device)
    n_total = 0
    t0 = time.time()
    for bi, batch in enumerate(text_batches(args.text_file, args.batch_size, args.max_texts)):
        E = encode_texts(batch, device, args.encode_batch)  # [B, d] normalized
        E = E.to(device)
        G += E.T @ E  # exact accumulation, [d, d]
        n_total += E.shape[0]
        elapsed = time.time() - t0
        rate = n_total / elapsed
        print(f"  batch {bi+1}: {n_total} texts, {rate:.0f} texts/s, {elapsed:.0f}s", flush=True)
    print(f"accumulated Gram over {n_total} texts")

    # ---- 2. Eigendecomposition -> V_T + spectrum ----
    evals, evecs = torch.linalg.eigh(G)
    evals = evals.flip(0)  # descending
    evecs = evecs.flip(1)  # columns descending
    V_T = evecs[:, : args.rank].contiguous()  # [d, rank]
    total_energy = evals.sum().item()
    kept_energy = evals[: args.rank].sum().item()
    print(f"\n[1] Anchor-corpus energy retention (r={args.rank}): {kept_energy/total_energy:.4f}")

    # ---- 3. Evaluation queries (same prompt convention as inference) ----
    eval_sets = {
        "scannet200": load_labels(SCANNET200_LABELS),
        "scannet20": load_labels(SCANNET20_LABELS),
        "lerf_ovs": get_lerf_ovs_labels(),
        "3dovs": get_3dovs_labels(),
    }
    report = {"rank": args.rank, "n_anchors": n_total,
              "anchor_energy_retention": kept_energy / total_energy}
    for name, labels in eval_sets.items():
        Q = encode_texts(labels, device, args.encode_batch, add_prefix=True)
        ret = energy_retention(Q, V_T)
        print(f"[2] {name:10s} ({len(labels):4d} queries): energy retention = {ret:.4f}")
        report[name] = ret

    # ---- 4. Category separability ----
    Q200 = encode_texts(eval_sets["scannet200"], device, args.encode_batch, add_prefix=True)
    q16 = Q200 @ V_T
    q16 = q16 / q16.norm(dim=-1, keepdim=True)
    sim_768 = (Q200 @ Q200.T).fill_diagonal_(0).abs().mean().item()
    sim_16 = (q16 @ q16.T).fill_diagonal_(0).abs().mean().item()
    print(f"[3] scannet200 mean pairwise |cos| : 768-dim {sim_768:.4f} -> 16-dim {sim_16:.4f}")
    report["scannet200_mean_pairwise_abs_cos_768"] = sim_768
    report["scannet200_mean_pairwise_abs_cos_16"] = sim_16

    # ---- 5. Save ----
    np.save(out_dir / "V_T.npy", V_T.float().cpu().numpy())
    np.save(out_dir / "gram.npy", G.float().cpu().numpy())
    np.save(out_dir / "singular_values.npy", torch.sqrt(evals.clamp(min=0)).cpu().numpy())
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_dir}/V_T.npy, gram.npy, singular_values.npy, report.json")

    ok = all(report[n] >= 0.8 for n in ["scannet200", "scannet20", "lerf_ovs", "3dovs"])
    print("\n" + "=" * 70)
    print(f"VERDICT: {'PASS (>=0.8 on all eval sets)' if ok else 'FAIL (<0.8 on some eval set)'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
