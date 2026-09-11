"""
Diag: input-information ceiling for the single-chunk L1 bottleneck.

Question: in the single-chunk deterministic overfit (00777c41d4_0), per-dim corr
plateaus at 0.2-0.6 with L1 ~0.4. Is that because (a) the per-point target is
only weakly predictable from the input attributes (information ceiling), or
(b) the transformer underfits (optimization/weighting)?

Method: train a plain MLP from point attributes + coordinates to the 16-dim
target on the SAME chunk, held-out per-dim corr = the ceiling a global function
of the inputs can reach. If the MLP reaches corr 0.7-0.9, the information IS in
the inputs and the transformer's failure is optimization/architecture.
If the MLP also stalls at ~0.3-0.5, the target's per-point values are genuinely
hard to predict from attributes -> the bottleneck is the target structure
(needs class-level supervision).

Variants: attrs+xyz / xyz-only / attrs-only.

Usage:
    /home/isom/.conda/envs/scene_splat/bin/python tools/diagnosis/diag_mlp_ceiling.py
"""
import numpy as np
import torch
import torch.nn as nn
import os

CHUNK = "/home/isom/cyf/SceneSplat/scannetpp_v2/train_grid1.0cm_chunk6x6_stride3x3/00777c41d4_0"
RANK = 16
SEED = 0
EPOCHS = 20
BATCH = 4096
LR = 1e-3
HID = 512


def load_chunk(path):
    d = np.load(os.path.join(path, f"lang_feat_grid_svd_r{RANK}.npz"))
    C = d["compressed"]; idx = d["indices"]
    vm = np.load(os.path.join(path, "valid_feat_mask.npy")).astype(bool)
    feat = C[idx]  # [V,16]
    attrs = []
    for k in ["color", "opacity", "quat", "scale"]:
        a = np.load(os.path.join(path, f"{k}.npy")).astype(np.float32)
        if a.ndim == 1:
            a = a[:, None]
        attrs.append(a[vm])
    xyz = np.load(os.path.join(path, "coord.npy")).astype(np.float32)[vm]
    X = np.concatenate([xyz] + attrs, axis=1)  # [V, 3+3+1+4+3=14]
    return X, feat


class MLP(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, HID), nn.ReLU(),
            nn.Linear(HID, HID), nn.ReLU(),
            nn.Linear(HID, HID), nn.ReLU(),
            nn.Linear(HID, dout),
        )

    def forward(self, x):
        return self.net(x)


def run_variant(X, T, label):
    rng = np.random.RandomState(SEED)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_tr = int(n * 0.8)
    tr, te = perm[:n_tr], perm[n_tr:]

    # normalize inputs on train stats
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd

    Xt = torch.from_numpy(Xn[tr]).float()
    Tt = torch.from_numpy(T[tr]).float()
    Xe = torch.from_numpy(Xn[te]).float()
    Te = torch.from_numpy(T[te]).float()

    model = MLP(Xn.shape[1], RANK)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.L1Loss()
    n_batches = (len(Xt) + BATCH - 1) // BATCH
    for ep in range(EPOCHS):
        model.train()
        perm_b = torch.randperm(len(Xt))
        tot = 0
        for i in range(n_batches):
            b = perm_b[i * BATCH:(i + 1) * BATCH]
            opt.zero_grad()
            out = model(Xt[b])
            loss = lossf(out, Tt[b])
            loss.backward()
            opt.step()
            tot += loss.item()
    model.eval()
    with torch.no_grad():
        Pe = model(Xe)
        # per-dim corr
        corrs = []
        for dd in range(RANK):
            p, t = Pe[:, dd].numpy(), Te[:, dd].numpy()
            if t.std() < 1e-9 or p.std() < 1e-9:
                corrs.append(0.0)
                continue
            corrs.append(np.corrcoef(p, t)[0, 1])
        corrs = np.array(corrs)
        l1 = torch.abs(Pe - Te).mean().item()
        # mean-removed cosine on waves (dims 1-15)
        pc, tc = Pe[:, 1:].numpy(), Te[:, 1:].numpy()
        pc = pc - pc.mean(0); tc = tc - tc.mean(0)
        cos = float((pc * tc).sum() / (np.linalg.norm(pc) * np.linalg.norm(tc) + 1e-9))
    print(f"[{label}] L1={l1:.4f} per-dim corr={np.round(corrs, 3)}")
    print(f"[{label}] dim1={corrs[1]:.3f} dim4={corrs[4]:.3f} minor_mean={np.mean(np.abs(corrs[1:])):.3f} "
          f"wave_mean-removed_cos={cos:.3f}")
    return corrs


def main():
    torch.manual_seed(SEED)
    X, T = load_chunk(CHUNK)
    print(f"chunk: {os.path.basename(CHUNK)} points={X.shape[0]} in_dims={X.shape[1]}")
    run_variant(X, T, "attrs+xyz (all 14)")
    run_variant(X[:, :3], T, "xyz-only")
    run_variant(X[:, 3:], T, "attrs-only (11)")


if __name__ == "__main__":
    main()
