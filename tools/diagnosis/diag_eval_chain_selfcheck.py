#!/usr/bin/env python3
"""
diag_f_vs_t.py — 诊断"模型预测特征 F 与 GT 压缩特征 T 到底差在哪"（2026-08-03）

三个模式（共用同一套 tester 流水线，只换模型/数据）：
  --mode gt_upper     : stub 模型输出 = canonicalized 压缩特征（T）→ 跑完整 tester 流水线
                        （Q 拟合 + sigmoid logits + topk + voting + metric）。
                        如果 ≈41.8% → 评测链 OK，F 是瓶颈；如果 ≈0 → 评测链仍有 bug。
                        （在 train chunk 场景上跑，因为只有那里有 npz）
  --mode train_fvst   : 真实模型在 train chunk 场景推理，捕获 F，与 npz 的 T 逐维对比
                        （corr / 幅度比 / 去均值 cos / PCA 谱）
  --mode val_capture  : 真实模型在 val 场景推理（只跑前 N 个），捕获 F/Q/text16/labels，
                        分析 Q 拟合秩、对齐后类均值 cos、逐点 argmax 分布（是否"每场景单类"）

用法（服务器）：
  cd /home/isom/cyf/CompressedSceneSplat
  /home/isom/.conda/envs/scene_splat/bin/python diag_f_vs_t.py \
      --mode val_capture --config configs/custom/lang-pretrain-ptv3m1-scannetpp-v2-smoke.py \
      --weight exp/smoke-ptv3m1-16-scannetpp-v2-invvar-contrast/model/model_best.pth \
      --save_path exp/diag_out/val_capture --n_scenes 4
"""
import argparse
import os
import numpy as np
import torch

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

from pointcept.engines.defaults import default_config_parser
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.utils.svd_sign import canonicalize_svd_sign
import pointcept.engines.test as test_mod
from tools.compute_procrustes_alignment_simple import compute_procrustes_Q_cuda_with_labels as _orig_Q

from pointcept.utils import comm  # noqa: E402


class SubsetDataset(torch.utils.data.Dataset):
    """只取前 n 个场景，避免 tester 跑全部 50 个 val 场景。"""
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = min(n, len(dataset))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]


class GTFStubModel(torch.nn.Module):
    """把 GT 压缩特征当作模型输出的 stub（2026-08-03 v2）。

    前向返回 dataset 加载的 lang_feat（GridSample 合并后的 cell 级特征）。
    注意：不能用 point_to_grid —— GridSample(mode=train) 会把 point_to_grid
    替换成自己的 cell 排序位置（与 npz indices 语义不同），导致 100% 错位。
    lang_feat 是 npz compressed[indices] 展开后经 FilterValidPoints/GridSample
    正确合并的值，与模型前向输出同一语义。
    """
    def __init__(self, current_name):
        super().__init__()
        self._current_name = current_name

    def forward(self, input_dict, chunk_size=None):
        scene_path = input_dict["scene_path"]
        scene_path = scene_path[0] if isinstance(scene_path, (list, tuple)) else scene_path
        self._current_name[0] = os.path.basename(str(scene_path))
        feat = input_dict["lang_feat"].float().cuda()
        return {"point_feat": {"feat": feat}}


def strip_module_prefix(state_dict):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def build_real_model(cfg, weight_path):
    model = build_model(cfg.model)
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    sd = strip_module_prefix(ckpt["state_dict"])
    missing, _ = model.load_state_dict(sd, strict=False)
    assert not missing, f"missing keys: {missing[:10]}"
    model.cuda().eval()
    print(f"[diag] loaded {weight_path} (epoch {ckpt.get('epoch')})")
    return model


def class_mean_rank(X, labels, n_classes=100):
    """复现 Q 拟合的输入结构：每点归一化 -> 类平均 -> SVD（有效秩）。"""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    valid = (labels >= 0) & (labels < n_classes)
    Xn, lb = Xn[valid], labels[valid]
    if len(lb) == 0:
        return None
    sum_j = np.zeros((n_classes, X.shape[1]))
    np.add.at(sum_j, lb, Xn)
    counts = np.bincount(lb, minlength=n_classes).astype(float).clip(min=1)
    sum_j = sum_j / counts[:, None]
    return np.linalg.svd(sum_j, full_matrices=False)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gt_upper", "train_fvst", "val_capture"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", default=None)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--n_scenes", type=int, default=4)
    args = ap.parse_args()

    cfg = default_config_parser(args.config, {})
    cfg.save_path = args.save_path
    os.makedirs(args.save_path, exist_ok=True)

    if args.mode in ("gt_upper", "train_fvst"):
        cfg.data.test.split = "train_grid1.0cm_chunk6x6_stride3x3"
        cfg.data.test.load_compressed_lang_feat = True   # 需要 point_to_grid
    else:
        cfg.data.test.load_compressed_lang_feat = False  # val 无 npz

    dataset = build_dataset(cfg.data.test)
    print(f"[diag] dataset len={len(dataset)} split={cfg.data.test.split}")
    subset = SubsetDataset(dataset, args.n_scenes)

    # ---- 捕获器：Q 拟合（per scene）----
    q_captures = []
    current_name = [None]

    def patched_Q(X_c, Y, labels):
        Q, metrics = _orig_Q(X_c, Y, labels)
        q_captures.append(dict(
            name=current_name[0],
            X_c=X_c.detach().cpu().numpy(),        # tester 传入的 accumulated_features[valid_mask]
            Y=Y.detach().cpu().numpy(),            # text16 [100, 16]
            labels=labels.detach().cpu().numpy(),  # [N_valid]
            Q=Q.detach().cpu().numpy(),
            metrics=metrics,
        ))
        return Q, metrics

    test_mod.compute_procrustes_Q_cuda_with_labels = patched_Q

    # ---- 构建模型 ----
    if args.mode == "gt_upper":
        model = GTFStubModel(current_name).cuda().eval()
    else:
        assert args.weight and os.path.isfile(args.weight), f"weight not found: {args.weight}"
        model = build_real_model(cfg, args.weight)

    # ---- 捕获器：模型输出（per fragment）----
    feat_captures = {}  # name -> dict(scene_path, pairs=[(index, feat)])
    orig_fwd = model.forward

    def wrapped_fwd(input_dict, **kwargs):
        out = orig_fwd(input_dict, **kwargs)
        scene_path = input_dict["scene_path"]
        scene_path = scene_path[0] if isinstance(scene_path, (list, tuple)) else scene_path
        name = os.path.basename(str(scene_path))
        current_name[0] = name
        idx = input_dict["index"].detach().cpu()
        feat = out["point_feat"]["feat"].detach().cpu().float()
        rec = feat_captures.setdefault(name, dict(scene_path=scene_path, pairs=[]))
        rec["pairs"].append((idx, feat))
        return out

    model.forward = wrapped_fwd

    # ---- 运行 tester ----
    test_loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True, collate_fn=test_mod.ZeroShotSemSegTester.collate_fn,
    )
    tester = test_mod.ZeroShotSemSegTester(cfg=cfg, model=model, test_loader=test_loader)
    tester.test()

    # ---- 重建 accumulated F 并保存 ----
    out_dir = args.save_path
    for cap in q_captures:
        sv = cap["metrics"]["singular_values"]
        print(f"[diag] scene {cap['name']}: Q-fit M_matrix singular values = {np.round(sv, 4)}")
        print(f"        det={cap['metrics']['det_Q']:.4f} cos {cap['metrics']['cosine_before']:.4f}->{cap['metrics']['cosine_after']:.4f} "
              f"N={cap['metrics']['N']}")
        rank_cm = class_mean_rank(cap["X_c"], cap["labels"])
        if rank_cm is not None:
            print(f"        class-mean(normalized-F) singular values = {np.round(rank_cm[:8], 4)}")

    for name, rec in feat_captures.items():
        all_idx = torch.cat([p[0] for p in rec["pairs"]])
        all_feat = torch.cat([p[1] for p in rec["pairs"]])
        N = int(all_idx.max()) + 1
        buf = torch.zeros(N, all_feat.shape[1])
        cnt = torch.zeros(N)
        buf.index_add_(0, all_idx, all_feat)
        cnt.index_add_(0, all_idx, torch.ones_like(all_idx.float()))
        valid = cnt > 0
        F_raw = buf[valid].numpy()
        F_idx = all_idx[valid].numpy()
        qcap = next((c for c in q_captures if c["name"] == name), None)
        np.savez(os.path.join(out_dir, f"{name}.npz"),
                 F_raw=F_raw, F_idx=F_idx,
                 Q=qcap["Q"] if qcap else np.eye(16),
                 text16=qcap["Y"] if qcap else np.zeros((0, 16)),
                 labels=qcap["labels"] if qcap else np.zeros(0, dtype=np.int64),
                 scene_path=rec["scene_path"])
        print(f"[diag] saved {name}.npz F_raw {F_raw.shape} scene_path={rec['scene_path']}")

    # ---- 逐点 argmax 分布（val_capture）----
    if args.mode == "val_capture":
        for cap in q_captures:
            name = cap["name"]
            data = np.load(os.path.join(out_dir, f"{name}.npz"))
            F_raw, Q, Y = data["F_raw"], data["Q"], data["text16"]
            F_align = F_raw @ Q
            logits = F_align @ Y.T
            pred = logits.argmax(1)
            uniq, counts = np.unique(pred, return_counts=True)
            order = np.argsort(-counts)
            print(f"\n[diag] scene {name} argmax distribution ({len(pred)} grid cells):")
            for i in order[:6]:
                cls = int(uniq[i])
                cname = tester.class_names[cls] if cls < len(tester.class_names) else "?"
                print(f"        class {cls} ({cname}): {counts[i]/len(pred)*100:.1f}%")
            Fn = F_raw / (np.linalg.norm(F_raw, axis=1, keepdims=True) + 1e-8)
            Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
            corr = Fn @ Yn.T
            print(f"        mean max-cos(F_align, text16): {corr.max(1).mean():.4f}")

    # ---- F vs T（train_fvst）----
    if args.mode == "train_fvst":
        for name, rec in feat_captures.items():
            npz_path = os.path.join(rec["scene_path"], "lang_feat_grid_svd_r16.npz")
            if not os.path.exists(npz_path):
                print(f"[diag] no npz for {name}, skip")
                continue
            d = np.load(npz_path)
            T_comp = canonicalize_svd_sign(d["compressed"].astype(np.float32))
            T_idx = d["indices"]
            data = np.load(os.path.join(out_dir, f"{name}.npz"))
            F_raw, F_idx = data["F_raw"], data["F_idx"]
            T_point = T_comp[T_idx[F_idx]]          # 每 cell 的 GT 特征
            ok = np.all(np.isfinite(T_point), axis=1) & (np.linalg.norm(T_point, axis=1) > 1e-8)
            F, T = F_raw[ok], T_point[ok]
            print(f"\n[diag] F-vs-T scene {name}: matched {F.shape[0]} cells")
            corr_dims = np.array([np.corrcoef(F[:, j], T[:, j])[0, 1] for j in range(16)])
            ratio_dims = np.abs(F).mean(0) / (np.abs(T).mean(0) + 1e-8)
            print(f"        per-dim corr : {np.round(corr_dims, 3)}")
            print(f"        |F|/|T| ratio: {np.round(ratio_dims, 3)}")
            Fn, Tn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8), T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
            print(f"        cos(F,T) mean: {np.sum(Fn * Tn, axis=1).mean():.4f}")
            Fd, Td = F - F.mean(0), T - T.mean(0)
            Fdn = Fd / (np.linalg.norm(Fd, axis=1, keepdims=True) + 1e-8)
            Tdn = Td / (np.linalg.norm(Td, axis=1, keepdims=True) + 1e-8)
            print(f"        cos(F-mean,T-mean) mean: {np.sum(Fdn * Tdn, axis=1).mean():.4f}")
            for nm, X in [("F", F), ("T", T)]:
                Xc = X - X.mean(0)
                _, S, _ = np.linalg.svd(Xc, full_matrices=False)
                var_frac = S ** 2 / (S ** 2).sum()
                print(f"        {nm} PCA var frac top5: {np.round(var_frac[:5], 3)} (top1={var_frac[0]:.3f})")


if __name__ == "__main__":
    main()
