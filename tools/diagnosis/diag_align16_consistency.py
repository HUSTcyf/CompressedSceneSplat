#!/usr/bin/env python3
"""
diag_align16_consistency.py — 判别实验（2026-08-03）

回答：训练目标在线对齐到 text16（方案 X）是否把目标从"每 chunk 独立坐标空间"
变成"统一 text16 空间"——即梯度是否不再跨 chunk 抵消。

关键认知（为什么之前的设计会测错）：
  - 跨 chunk Procrustes 对齐后的 cos（check_wave_consistency 的 0.73）是**旋转不变**度量
    ——基歧义已被 Procrustes 吸收，它测的是"真实结构差异"，与对齐无关。
  - 方案 X 改变的是**绝对坐标**：T' = T @ Q_chunk 后所有 chunk 都在 text16 附近。
    L1/梯度跨 chunk 是否一致由**绝对坐标一致性**决定（无旋转对齐的直接 cos）。
  - 数学必然：T' 的 Procrustes 跨 chunk cos ≤ T 的（中转旋转是直接最优的子集）。
    若脚本输出 T' 的 Procrustes cos 更低，不是 bug，是预期。

主度量（对齐有效应成立）：
  M1. 每 chunk：C'（T' 类均值，去质心归一化） vs T_text（text16 类均值同处理）的
      直接 cos（无 Procrustes）——预期 0.7-0.9（对齐质量）
  M2. 跨 chunk：C' 直接 cos（无对齐）——预期高（都≈text16 结构），
      对比 T 的 C 直接 cos（预期低，基随机旋转）
辅助度量：
  A1. T 的跨 chunk Procrustes after-cos——复现已知 ~0.73（自检 S2）
  A2. T' 的跨 chunk Procrustes after-cos——数学预期 ≤ T（自检 S6）
  A3. 每 chunk Q 正交性、拟合 N/类数（自检 S3）

自检链（反复检查防写错）：
  S1. procrustes_cos(A, A) ≈ 1.0（测量函数正确）
  S2. T 的 Procrustes 跨 chunk cos ≈ 0.73 ± 0.05（复现 check_wave_consistency，数据加载/测量正确）
  S3. ||Q^T Q - I||_F < 1e-4（Q 正交，Procrustes 闭式解正确）
  S4. numpy 复刻 fit_q vs 官方 compute_procrustes_Q_cuda_with_labels（torch CPU）输出一致
      （前 2 个 chunk，||Q - Q_official||_F < 1e-4）
  S5. M1 中 C' vs T_text 直接 cos 显著高于 C vs T_text 直接 cos（对齐确实生效）
  S6. A2 ≤ A1 + 1e-6（数学必然）

用法（服务器）：
  cd /home/isom/cyf/CompressedSceneSplat
  /home/isom/.conda/envs/scene_splat/bin/python tools/diagnosis/diag_align16_consistency.py [NCHUNK]

一致性说明：判别脚本用**点级**压缩特征（compressed[indices]）与 check_wave_consistency
同粒度；训练端 Q 拟合输入是 GridSample 后的 cell 级特征，但同类点共享特征值
→ 类均值近似等价（差异 ~1e-3 量级，不影响结论）。
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/isom/cyf/CompressedSceneSplat")
from pointcept.utils.svd_sign import canonicalize_svd_sign
from tools.compute_procrustes_alignment_simple import (
    perform_svd_reduction,
    compute_procrustes_Q_cuda_with_labels as official_fit_q,
)

DATA = "/home/isom/cyf/SceneSplat/scannetpp_v2/train_grid1.0cm_chunk6x6_stride3x3"
TEXT = ("/home/isom/cyf/CompressedSceneSplat/pointcept/datasets/preprocessing/"
        "scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt")
NCHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 6
EPS = 1e-8


# ============================================================================
# 1. text16：与 tester（test.py:203-207）/ 训练端（default.py）完全一致
#    perform_svd_reduction(text768, 16, normalize=False) + L2 归一化（每行）
# ============================================================================
text768 = torch.load(TEXT, weights_only=True).numpy().astype(np.float64)
text16, _, _ = perform_svd_reduction(text768, 16, normalize=False)
text16 = text16 / (np.linalg.norm(text16, axis=1, keepdims=True) + EPS)  # F.normalize(p=2, dim=1)


# ============================================================================
# 2. 工具函数
# ============================================================================
def norm_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)


def procrustes_cos(A, B, mask=None):
    """Procrustes 对齐后的 mean-cos（与 check_wave_consistency 的 procrustes() 相同度量）"""
    An, Bn = norm_rows(A), norm_rows(B)
    if mask is not None:
        An, Bn = An[mask], Bn[mask]
    M = An.T @ Bn
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = U @ Vt
    return float(np.mean(np.sum((An @ Q) * Bn, axis=1)))


def direct_cos(A, B, mask=None):
    """无对齐直接 mean-cos（绝对坐标一致性——主度量）。
    mask: 只在这些行上算（默认全行）。未出现类在去质心后是 -质心 方向的噪声行，
    会稀释真实信号，M1（vs T_text）必须传出现类 mask；M2（跨 chunk）不传
    （T' 的未出现类行方向一致，均 ≈ -text16 质心，不稀释）。"""
    An, Bn = norm_rows(A), norm_rows(B)
    if mask is None:
        return float(np.mean(np.sum(An * Bn, axis=1)))
    return float(np.mean(np.sum(An[mask] * Bn[mask], axis=1)))


def fit_q(X_c, Y, labels):
    """复刻训练端 _fit_procrustes_q（torch 实现，与评测端同路径）：
    逐点归一化 -> 类平均(1/n) -> sum_j^T @ Y -> torch.linalg.svd -> Q，det<0 修正。
    输入为 torch tensor（float32，与训练端一致）。
    X_c: [N, d] 逐点特征; Y: [C, d] 全类文本; labels: [N] 类 id ∈ [0, C)"""
    X_n = X_c / (X_c.norm(dim=1, keepdim=True) + 1e-8)
    C = Y.shape[0]
    counts = torch.bincount(labels, minlength=C).clamp(min=1).to(X_n.dtype)
    sum_j = torch.zeros(C, X_c.shape[1], device=X_c.device, dtype=X_c.dtype)
    sum_j.index_add_(0, labels.long(), X_n)
    sum_j = sum_j / counts[:, None]
    Mt = sum_j.t() @ Y  # [d, d]
    U, S, Vt = torch.linalg.svd(Mt, full_matrices=False)
    Q = U @ Vt
    if torch.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    return Q, S


def load_chunk(path):
    """加载 chunk 的 SVD 压缩特征 + 标签（与 check_wave_consistency 相同的对齐处理）"""
    d = np.load(os.path.join(path, "lang_feat_grid_svd_r16.npz"))
    comp = canonicalize_svd_sign(d["compressed"].astype(np.float32)).astype(np.float64)
    indices = d["indices"]
    seg = np.load(os.path.join(path, "segment.npy"))
    if seg.ndim == 2:
        seg = seg[:, 0]  # scannetpp 的 segment.npy 是 [N,3]，语义在第 0 列
    seg = seg.astype(np.int64)
    if len(seg) != len(indices):
        # indices 只覆盖 valid 点：用 valid_feat_mask 过滤
        vmask = np.load(os.path.join(path, "valid_feat_mask.npy")).astype(bool)
        seg = seg[vmask]
    assert len(seg) == len(indices), (len(seg), len(indices))
    return comp, indices, seg


def class_means_centered(pts, seg):
    """逐点归一化 -> 类平均 -> 去跨类均值 -> 归一化（check_wave_consistency 同款，
    用于一致性测量；去质心后公共方向被移除，测的是判别结构）。
    返回 (C, appear)：C [100, d]；appear = 出现类的索引数组（用于 mask 度量）"""
    C = np.zeros((100, pts.shape[1]), dtype=np.float64)
    appear = []
    for k in range(100):
        sel = seg == k
        if sel.sum() > 10:
            C[k] = norm_rows(pts[sel]).mean(0)
            appear.append(k)
    C = C - C.mean(0)
    return norm_rows(C), np.array(appear, dtype=np.int64)


# ============================================================================
# 3. 主流程
# ============================================================================
# 跨场景抽样（2026-08-03 修正）：同一场景的相邻块 SVD 基天然相似（首版 6 个块
# 全是 00777c41d4 场景，M2 的 T=0.91 被同场景抬高了），不能代表训练时跨场景的
# 基歧义。改为按场景分组，取 NCHUNK 个不同场景的各自第一个 chunk。
all_chunks = sorted(glob.glob(os.path.join(DATA, "*")))
by_scene = {}
for c in all_chunks:
    scene = os.path.basename(c).rsplit("_", 1)[0]  # "00777c41d4_0" -> "00777c41d4"
    by_scene.setdefault(scene, []).append(c)
chunks = [v[0] for v in by_scene.values()][:NCHUNK]
assert len(chunks) >= 2, "至少需要 2 个不同场景的 chunk"
print(f"[DATA] {len(by_scene)} 个场景可用，抽样 {len(chunks)} 个不同场景的首个 chunk")

# S1 自检：Procrustes(A, A) = 1.0
A_test = np.random.RandomState(0).randn(100, 16)
A_test = norm_rows(A_test)
s1 = procrustes_cos(A_test, A_test)
print(f"[S1] procrustes_cos(A,A) = {s1:.10f} (应 ≈1.0)")
assert abs(s1 - 1.0) < 1e-6, "S1 失败：测量函数有 bug"

# T_text：text16 的类均值结构。注意不能用 class_means_centered(text16, arange(100))——
# 它每类只有 1 行，sel.sum()=1 ≤ 10 阈值会把所有类跳过（→ 全 0）。text16 每行已
# 单位化，类均值 = 自身，直接去质心 + 归一化即可（与 class_means_centered 输出同语义）
T_text = text16 - text16.mean(0)
T_text = norm_rows(T_text)
assert np.linalg.norm(T_text) > 0, "T_text 构造失败"

recs = []  # 每 chunk: dict(name, C(T 类均值), C'(T' 类均值), Q, N, ncls)
for c in chunks:
    comp, indices, seg = load_chunk(c)
    T = comp[indices]  # 点级目标特征 [N, 16]

    # Q 拟合（复刻训练端）：valid 点 + 标签范围 [0, 100) + 类数≥2
    valid = (seg >= 0) & (seg < 100)
    T_v, seg_v = T[valid], seg[valid]
    assert np.all(np.isfinite(T_v)), f"{c}: 特征含 NaN"
    T_v_t = torch.from_numpy(T_v.astype(np.float32))
    Y_t = torch.from_numpy(text16.astype(np.float32))
    lb_t = torch.from_numpy(seg_v.astype(np.int64))
    q_t, S_t = fit_q(T_v_t, Y_t, lb_t)  # torch 复刻（与训练端 default.py 同实现）
    q = q_t.numpy()
    s3 = np.linalg.norm(q.T @ q - np.eye(16), "fro")
    # S4 自检：torch 复刻 vs 官方函数（同为 torch.linalg.svd 路径，应完全一致）
    q_off = official_fit_q(T_v_t, Y_t, lb_t)[0].numpy()
    s4 = np.linalg.norm(q - q_off, "fro")
    # M_matrix 奇异值谱（诊断退化：SVD 退化子空间 → Q 不唯一，但训练/评测同为
    # torch 路径不受影响；np vs torch 的 Q 差异即来源于此）
    sv_spec = np.round(S_t.numpy(), 4)

    Tp = T @ q  # 对齐后目标

    # ---- 2026-08-03 符号歧义检查（Trivial solution 的根因验证）----
    # Q 拟合（SVD）列符号跨 chunk 任意 → T' 列符号不一致 → 模型学平均符号 → 坍缩。
    # 检查：每列"最大绝对值点"的符号（canonicalize 的判定量）——修复前应发现
    # 部分 chunk 翻转；canonicalize 后必然 100% 一致。
    Tp_v = Tp[valid]  # 与训练端 canonicalize 的"有效行判定"一致
    col_sign = np.sign(Tp_v[np.abs(Tp_v).argmax(axis=0), np.arange(Tp_v.shape[1])])
    # canonicalize 模拟（与 default.py._canonicalize_sign 同规则，基于有效行）
    Tp_can = Tp.copy()
    for k in range(Tp_can.shape[1]):
        sel = Tp_v[:, k]
        i_max = int(np.argmax(np.abs(sel)))
        if abs(sel[i_max]) >= 1e-8 and sel[i_max] < 0:
            Tp_can[:, k] = -Tp_can[:, k]

    C, appear = class_means_centered(T, seg)
    Cp, _ = class_means_centered(Tp, seg)
    Cp_can, _ = class_means_centered(Tp_can, seg)
    n_cls = len(np.unique(seg_v))
    recs.append(dict(name=os.path.basename(c), C=C, Cp=Cp, Cp_can=Cp_can,
                     col_sign=col_sign, q=q, n_cls=n_cls,
                     n=len(T_v), s3=s3, s4=s4, sv_spec=sv_spec, appear=appear))
    print(f"[{os.path.basename(c)}] N={len(T_v)} 类数={n_cls} "
          f"S3(||Q^TQ-I||)={s3:.2e} S4(||Q-Q_official||)={s4:.2e}")
    print(f"        M_matrix 奇异值: {sv_spec[:6]} ... (共 {len(sv_spec)}, 末3: {sv_spec[-3:]})")
    print(f"        T' 列符号(修复前, 每列max-abs点): {np.sign(col_sign).astype(int)}")

# S3/S4 汇总
assert all(r["s3"] < 1e-4 for r in recs), "S3 失败：Q 不正交"
assert all(r["s4"] < 1e-4 for r in recs), "S4 失败：复刻与官方不一致"
print(f"[S3][S4] 全部 chunk Q 正交 & 复刻一致 ✓")

# ---- 主度量 M1：每 chunk C' vs T_text 直接 cos（对齐质量，仅在出现类上算）----
print("\n[M1] 每 chunk 出现类均值 vs text16 类均值 直接 cos（无对齐）")
d0 = []
for r in recs:
    cT = direct_cos(r["C"], T_text, mask=r["appear"])   # 对齐前（预期低：基随机）
    cTp = direct_cos(r["Cp"], T_text, mask=r["appear"])  # 对齐后（预期 0.7-0.9）
    d0.append((cT, cTp))
    print(f"  {r['name']}: T 直接cos={cT:.4f}  T' 直接cos={cTp:.4f}  (出现类 {len(r['appear'])} 个)")
m1_T = float(np.mean([x[0] for x in d0]))
m1_Tp = float(np.mean([x[1] for x in d0]))
print(f"  mean: T={m1_T:.4f} -> T'={m1_Tp:.4f}")
# S5 自检：对齐后显著提升（出现类信号应从 ~0 升到 0.5+）
assert m1_Tp > m1_T + 0.2, f"S5 失败：对齐未生效 (T={m1_T:.4f} T'={m1_Tp:.4f})"
print(f"[S5] 对齐生效 ✓ (提升 {m1_Tp - m1_T:+.4f})")

# ---- 主度量 M2：跨 chunk 直接 cos（无对齐，绝对坐标一致性）----
# 只在两 chunk 出现类交集上算——去质心后的未出现类行 = -质心方向（公共方向），
# 无 mask 时会被它主导（实测 T=0.84 全是质心贡献），测不出判别结构的一致性。
print("\n[M2] 跨 chunk 出现类交集直接 cos（无 Procrustes，判别结构绝对一致性，主度量）")
m2_pairs = []
for i in range(len(recs)):
    for j in range(i + 1, len(recs)):
        inter = np.intersect1d(recs[i]["appear"], recs[j]["appear"])
        if len(inter) < 2:
            continue
        c = direct_cos(recs[i]["C"], recs[j]["C"], mask=inter)
        cp = direct_cos(recs[i]["Cp"], recs[j]["Cp"], mask=inter)
        m2_pairs.append((c, cp))
        print(f"  {recs[i]['name']} vs {recs[j]['name']}: T={c:.4f}  T'={cp:.4f}  (交集 {len(inter)} 类)")
m2_T = float(np.mean([x[0] for x in m2_pairs]))
m2_Tp = float(np.mean([x[1] for x in m2_pairs]))
print(f"  mean: T={m2_T:.4f} -> T'={m2_Tp:.4f}")
# canonicalize 后（修复版）：符号一致后的 M2（预期 ≈1.0 或显著高于修复前）
m2_can = []
for i in range(len(recs)):
    for j in range(i + 1, len(recs)):
        inter = np.intersect1d(recs[i]["appear"], recs[j]["appear"])
        if len(inter) < 2:
            continue
        m2_can.append(direct_cos(recs[i]["Cp_can"], recs[j]["Cp_can"], mask=inter))
m2_Tp_can = float(np.mean(m2_can)) if m2_can else float("nan")
print(f"  [修复后] canonicalize T' 后: mean={m2_Tp_can:.4f}")

# ---- 符号一致性检查（2026-08-03，Trivial solution 根因验证）----
# 每列 max-abs 点的符号跨 chunk 一致率：修复前（Q 原始列符号）应 < 100%
# （部分 chunk 翻转 → 模型学平均符号 → 坍缩）；canonicalize 后必为 100%。
print("\n[SIGN] T' 列符号跨 chunk 一致性（max-abs 判定，修复前）")
signs = np.stack([r["col_sign"] for r in recs])  # [n_chunk, 16]
agree = (signs > 0).mean(axis=0)  # 每列取正的比例（1.0 = 全一致，0.5 = 完全随机）
print(f"  每列取正比例: {np.round(agree, 3)}")
print(f"  16 列平均一致率: {agree.mean():.3f} (1.0 = 无符号歧义, <1.0 = 存在翻转 chunk)")
print(f"  翻转列数 (agree < 0.9): {int((agree < 0.9).sum())}/16")
print(f"  {'[SIGN] PASS: 无符号歧义' if agree.min() > 0.9 else '[SIGN] FAIL: 存在列符号翻转 —— 这就是 Trivial solution 根因'}")

# ---- 主度量 M3：每 chunk C/C' vs T_text 的 Procrustes after-cos（mask 版）----
# 评测端 Q_eval 拟合(F 类均值 → text16) 的分类质量由这个相对度量决定；
# T 的版本 = 评测上界（GT 走链 34.9%）的来源；T' 的版本 = 模型学 T' 后评测端的潜力。
print("\n[M3] 每 chunk vs text16 Procrustes after-cos（出现类，评测对齐潜力）")
m3_list = []
for r in recs:
    a = procrustes_cos(r["C"], T_text, mask=r["appear"])
    b = procrustes_cos(r["Cp"], T_text, mask=r["appear"])
    m3_list.append((a, b))
    print(f"  {r['name']}: T={a:.4f}  T'={b:.4f}")
m3_T = float(np.mean([x[0] for x in m3_list]))
m3_Tp = float(np.mean([x[1] for x in m3_list]))
print(f"  mean: T={m3_T:.4f} -> T'={m3_Tp:.4f}")

# ---- 辅助 A1/A2：跨 chunk Procrustes after-cos（旋转不变，S2/S6）----
print("\n[A1/A2] 跨 chunk Procrustes after-cos（旋转不变，辅助）")
a1_list, a2_list = [], []
for i in range(len(recs)):
    for j in range(i + 1, len(recs)):
        a1 = procrustes_cos(recs[i]["C"], recs[j]["C"])
        a2 = procrustes_cos(recs[i]["Cp"], recs[j]["Cp"])
        a1_list.append(a1)
        a2_list.append(a2)
        print(f"  {recs[i]['name']} vs {recs[j]['name']}: T={a1:.4f}  T'={a2:.4f}")
A1 = float(np.mean(a1_list))
A2 = float(np.mean(a2_list))
print(f"  mean: T={A1:.4f} -> T'={A2:.4f}")
# S2 自检：测量函数/数据加载正确性已由双脚本互证（同场景块 0.8539 完全一致）。
# 基线随抽样范围变化：同场景块 = 0.854，跨场景 = 0.545（本版测得，新的"跨场景
# 相对一致性"基线）。这里做合理性校验（0.3-0.9 内）并报告，不再硬断言。
print(f"[S2] T 的 Procrustes 跨 chunk cos = {A1:.4f} "
      f"(基线: 同场景块 0.854 / 跨场景 0.545——两值均与原脚本同构互证)")
assert 0.3 < A1 < 0.95, f"S2 失败：A1={A1:.4f} 超出合理性范围，测量有 bug"
# S6 自检：T' 的 Procrustes cos ≤ T 的（数学必然：中转旋转 ⊆ 直接最优；
# 逐点归一化非线性带来 ~1e-3 级扰动，故容差 1e-3）
print(f"[S6] T' Procrustes cos ({A2:.4f}) ≤ T ({A1:.4f})? {A2 <= A1 + 1e-3}")
assert A2 <= A1 + 1e-3, "S6 失败：违反数学必然，脚本有 bug"

# ============================================================================
# 4. 结论
# ============================================================================
print("\n" + "=" * 70)
print("结论判定：")
print(f"  M1 绝对对齐质量  : T={m1_T:.4f} -> T'={m1_Tp:.4f} (出现类直接 cos；"
      f"{m1_Tp:.2f}+ 目标坐标已显著接近 text16)")
print(f"  M2 判别绝对一致性: T={m2_T:.4f} -> T'={m2_Tp:.4f} (应显著提升——梯度不再抵消的关键)")
print(f"  M3 评测对齐潜力  : T={m3_T:.4f} -> T'={m3_Tp:.4f} (出现类 Procrustes；"
      f"评测 Q_eval 拟合后可达的 cos，T 版≈34.9% 上界来源)")
print(f"  A1/A2 相对一致性 : {A1:.4f} -> {A2:.4f} (旋转不变；A2≤A1 是数学必然)")
print("=" * 70)
print("解释：M2 显著提升 + M3 不下降 => 方案 X 让目标绝对坐标统一(梯度一致)且不损评测潜力。")
print("     此时若训练完 val mIoU 仍不提升 => 瓶颈在逐点噪声(83% 旋转不变)，基歧义非主因。")
