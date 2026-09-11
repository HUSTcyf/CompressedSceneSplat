"""SVD 压缩特征符号规范化（加载时执行，磁盘数据保持原始）。

原理：
    compressed 的第 k 列 = 网格特征在基向量 v_k 上的坐标（compressed = X·V）。
    翻转 v_k 的符号 ⟺ 翻转 compressed 第 k 列全体元素符号。
    因此对每列做"最大绝对值分量取正"（scikit-learn PCA.sign_flip 的等价做法），
    把每个场景的 SVD 基符号规范为全局一致，消除逐场景基的符号歧义。

    训练观测：~11.7% 场景的模型输出在 ±target 间随机翻转（cos_loss 从 0.05 跳到
    1.0/1.94），根源是任意符号的目标让模型无从学到一致约定。规范化后符号规则
    由数据确定（每列最大系数为正），模型可从输入推断并学会。

为什么在加载时做而不是改磁盘：
    - 磁盘 npz 保持原始/可逆，将来想换约定改代码即可
    - 操作是确定性的（不依赖模型/训练状态），训练与评测看到的约定一致
    - 计算开销可忽略（每列一次 argmax + 至多一次整列取反）

与离线工具 tools/compression/canonicalize_svd_sign.py 的逻辑保持一致。
"""
import numpy as np


def canonicalize_svd_sign(compressed: np.ndarray) -> np.ndarray:
    """对每列做最大绝对值取正，返回新数组（不原地修改输入）。

    Args:
        compressed: [M, D] 压缩特征（每列一个 SVD 基方向的坐标）

    Returns:
        规范化后的数组：每列最大绝对值分量均为正
    """
    out = compressed.copy()
    D = out.shape[1]
    for k in range(D):
        col = out[:, k]
        i_max = int(np.argmax(np.abs(col)))
        if abs(col[i_max]) < 1e-8:
            continue  # 全零列（理论罕见），跳过保持原样
        if col[i_max] < 0:
            out[:, k] = -col
    return out


def remove_scene_mean(compressed: np.ndarray) -> np.ndarray:
    """每场景去均值：移除场景偏移（公共方向），使训练目标聚焦判别波动。

    背景（2026-08-03 调研）：
    - 压缩坐标 = 场景偏移 μ（占主导）+ 语义波动 S + 噪声
    - cos/L1 损失的梯度被 μ 主导 → 模型只学到公共方向（波动每维 corr ≈ 0）
    - 去均值后损失聚焦 S → 波动可学
    - 代价：μ 中结构类（wall/floor/ceiling）的判别信息丢失（上界 44%→31%）

    评测对应：模型输出减输出均值（无监督，推理时无需 GT），text 减类均值。
    """
    return compressed - compressed.mean(axis=0)
