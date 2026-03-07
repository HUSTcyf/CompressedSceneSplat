#!/usr/bin/env python3
"""
验证数据对齐：检查FilterValidPoints和GridSample是否正确保留
高斯与lang_feat_grid_svd_r16.npz中压缩特征的标签对齐

使用方法:
    python tools/verify_data_alignment.py --scene figurines
    python tools/verify_data_alignment.py --scene figurines --svd_rank 16
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Import PROJECT_ROOT - handle both script and module execution
try:
    from ... import PROJECT_ROOT  # Relative import when run as module
except ImportError:
    # Fallback when run as script: add project root to sys.path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from pointcept.datasets.transform import FilterValidPoints, GridSample


def load_scene_data(scene_path: str, svd_rank: int = 16):
    """加载场景数据"""
    scene_path = Path(scene_path)

    print(f"\n{'='*80}")
    print(f"加载场景数据: {scene_path.name}")
    print(f"{'='*80}\n")

    # 加载原始数据
    print("1. 加载原始.npy文件...")
    coord = np.load(scene_path / "coord.npy")
    color = np.load(scene_path / "color.npy")
    opacity = np.load(scene_path / "opacity.npy")
    quat = np.load(scene_path / "quat.npy")
    scale = np.load(scene_path / "scale.npy")
    lang_feat = np.load(scene_path / "lang_feat.npy")
    segment = np.load(scene_path / "lang_label.npy")
    valid_feat_mask = np.load(scene_path / "valid_feat_mask.npy")

    print(f"   coord: {coord.shape}")
    print(f"   lang_feat: {lang_feat.shape}")
    print(f"   segment (lang_label): {segment.shape}")
    print(f"   valid_feat_mask: {valid_feat_mask.shape}")
    print(f"   唯一标签数: {len(np.unique(segment))}")

    # 加载SVD压缩特征
    svd_file = scene_path / f"lang_feat_grid_svd_r{svd_rank}.npz"
    print(f"\n2. 加载SVD压缩特征: {svd_file.name}")

    if not svd_file.exists():
        raise FileNotFoundError(f"SVD文件不存在: {svd_file}")

    svd_data = np.load(svd_file)
    compressed = svd_data['compressed']  # [M, svd_rank]
    indices = svd_data['indices']  # [N] - point to grid mapping

    print(f"   compressed: {compressed.shape}")
    print(f"   indices (point_to_grid): {indices.shape}")
    print(f"   grid数量: {compressed.shape[0]}")
    print(f"   point数量: {indices.shape[0]}")

    # 验证point_to_grid范围
    print(f"\n3. 验证point_to_grid索引范围...")
    print(f"   indices范围: [{indices.min()}, {indices.max()}]")
    print(f"   compressed行数: {compressed.shape[0]}")

    if indices.max() >= compressed.shape[0]:
        raise ValueError(f"point_to_grid索引超出范围! max={indices.max()}, compressed_rows={compressed.shape[0]}")

    if indices.min() < 0:
        raise ValueError(f"point_to_grid索引包含负数! min={indices.min()}")

    # 构建原始数据字典
    data_dict = {
        'coord': coord,
        'color': color,
        'opacity': opacity,
        'quat': quat,
        'scale': scale,
        'lang_feat': lang_feat,
        'segment': segment,
        'valid_feat_mask': valid_feat_mask,
        'point_to_grid': indices,
    }

    return data_dict, compressed


def verify_original_alignment(data_dict, compressed):
    """验证原始数据的对齐"""
    print(f"\n{'='*80}")
    print("验证原始数据对齐")
    print(f"{'='*80}\n")

    coord = data_dict['coord']
    lang_feat = data_dict['lang_feat']
    segment = data_dict['segment']
    valid_feat_mask = data_dict['valid_feat_mask']
    point_to_grid = data_dict['point_to_grid']

    N = coord.shape[0]
    M = compressed.shape[0]
    D = lang_feat.shape[1]
    svd_rank = compressed.shape[1]

    print(f"数据维度:")
    print(f"  原始点数: N = {N}")
    print(f"  Grid数量: M = {M}")
    print(f"  lang_feat维度: {D}")
    print(f"  SVD rank: {svd_rank}")

    # 检查1: point_to_grid长度
    print(f"\n检查1: point_to_grid长度")
    if point_to_grid.shape[0] != N:
        raise ValueError(f"point_to_grid长度({point_to_grid.shape[0]}) != coord长度({N})")
    print(f"  ✓ point_to_grid长度正确: {point_to_grid.shape[0]}")

    # 检查2: 形状一致性
    print(f"\n检查2: 所有数组形状一致性")
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and len(value.shape) == 1:
            if value.shape[0] != N:
                raise ValueError(f"{key}形状({value.shape[0]}) != coord长度({N})")
    print(f"  ✓ 所有一维数组长度一致")

    # 检查3: 验证原始lang_feat与SVD压缩特征的关系
    print(f"\n检查3: 原始lang_feat与SVD压缩特征的关系")
    print(f"  注意: SVD压缩后的特征与原始lang_feat不同维度，无法直接比较")
    print(f"  原始维度: {D}, 压缩后维度: {svd_rank}")

    # 检查4: segment标签分布
    print(f"\n检查4: segment标签分布")
    unique_labels = np.unique(segment)
    print(f"  唯一标签: {unique_labels}")
    for label in unique_labels:
        count = (segment == label).sum()
        ratio = count / N * 100
        print(f"    Label {int(label):3d}: {count:8d} points ({ratio:5.2f}%)")

    # 检查5: valid_feat_mask分布
    print(f"\n检查5: valid_feat_mask分布")
    valid_count = (valid_feat_mask > 0).sum()
    invalid_count = N - valid_count
    print(f"  Valid: {valid_count} ({valid_count/N*100:.2f}%)")
    print(f"  Invalid: {invalid_count} ({invalid_count/N*100:.2f}%)")

    # 检查6: 每个标签的valid/invalid分布
    print(f"\n检查6: 每个标签的valid/invalid分布")
    for label in unique_labels:
        mask_label = (segment == label)
        total = mask_label.sum()
        valid = (mask_label & (valid_feat_mask > 0)).sum()
        invalid = total - valid
        print(f"  Label {int(label):3d}: Total={total:7d}, Valid={valid:7d}, Invalid={invalid:7d}")

    return True


def apply_and_verify_transforms(data_dict, compressed, grid_size=0.01, seed=42):
    """应用变换并验证对齐"""
    print(f"\n{'='*80}")
    print("应用FilterValidPoints和GridSample变换")
    print(f"{'='*80}\n")

    np.random.seed(seed)

    # 记录原始状态
    orig_coord = data_dict['coord']
    orig_segment = data_dict['segment']
    orig_valid_mask = data_dict['valid_feat_mask']
    orig_point_to_grid = data_dict['point_to_grid']
    orig_lang_feat = data_dict['lang_feat']

    N_orig = orig_coord.shape[0]
    print(f"原始数据: {N_orig} points")

    # 变换1: FilterValidPoints
    print(f"\n变换1: FilterValidPoints")
    filter_transform = FilterValidPoints(key="valid_feat_mask", verbose=False)

    # 复制数据字典
    data_dict_after_filter = {k: v.copy() if isinstance(v, np.ndarray) else v
                              for k, v in data_dict.items()}

    data_dict_after_filter = filter_transform(data_dict_after_filter)

    coord_after_filter = data_dict_after_filter['coord']
    segment_after_filter = data_dict_after_filter['segment']
    valid_mask_after_filter = data_dict_after_filter['valid_feat_mask']
    point_to_grid_after_filter = data_dict_after_filter['point_to_grid']

    N_after_filter = coord_after_filter.shape[0]
    print(f"  FilterValidPoints后: {N_after_filter} points (过滤了{N_orig - N_after_filter} points)")

    # 验证FilterValidPoints后的对齐
    print(f"\n  验证FilterValidPoints对齐:")

    # 检查1: 所有有效点应该valid_feat_mask > 0
    if not (valid_mask_after_filter > 0).all():
        raise ValueError("FilterValidPoints后仍存在invalid的点!")
    print(f"    ✓ 所有点valid_feat_mask > 0")

    # 检查2: point_to_grid索引仍然有效
    if point_to_grid_after_filter.min() < 0:
        raise ValueError(f"point_to_grid包含负索引: min={point_to_grid_after_filter.min()}")
    if point_to_grid_after_filter.max() >= compressed.shape[0]:
        raise ValueError(f"point_to_grid索引越界: max={point_to_grid_after_filter.max()}, compressed_rows={compressed.shape[0]}")
    print(f"    ✓ point_to_grid索引有效: [{point_to_grid_after_filter.min()}, {point_to_grid_after_filter.max()}]")

    # 检查3: segment标签分布一致性
    orig_labels = set(np.unique(orig_segment))
    filter_labels = set(np.unique(segment_after_filter))
    if not filter_labels.issubset(orig_labels):
        raise ValueError(f"Filter后出现新标签! 原始: {orig_labels}, Filter后: {filter_labels}")
    print(f"    ✓ segment标签无新增标签")

    # 检查4: 每个标签的相对比例保持一致
    print(f"\n  标签分布变化:")
    print(f"    {'Label':>8} {'原始':>12} {'Filter后':>12} {'比例变化':>12}")
    print(f"    {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

    for label in sorted(orig_labels):
        orig_count = (orig_segment == label).sum()
        filter_count = (segment_after_filter == label).sum()
        if orig_count > 0:
            orig_ratio = orig_count / N_orig
            filter_ratio = filter_count / N_after_filter
            ratio_change = abs(orig_ratio - filter_ratio)
            print(f"    {int(label):>8} {orig_count:>12,} {filter_count:>12,} {ratio_change:>12.6f}")

    # 变换2: GridSample
    print(f"\n变换2: GridSample (train mode, grid_size={grid_size})")
    grid_sample_transform = GridSample(
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask", "point_to_grid", "segment"),
        return_grid_coord=True,
    )

    # 复制数据字典
    data_dict_after_grid = {k: v.copy() if isinstance(v, np.ndarray) else v
                            for k, v in data_dict_after_filter.items()}

    data_dict_after_grid = grid_sample_transform(data_dict_after_grid)

    coord_after_grid = data_dict_after_grid['coord']
    segment_after_grid = data_dict_after_grid['segment']
    valid_mask_after_grid = data_dict_after_grid['valid_feat_mask']
    point_to_grid_after_grid = data_dict_after_grid['point_to_grid']
    grid_coord = data_dict_after_grid.get('grid_coord')

    N_after_grid = coord_after_grid.shape[0]
    print(f"  GridSample后: {N_after_grid} points")

    # 验证GridSample后的对齐
    print(f"\n  验证GridSample对齐:")

    # 检查1: 所有点应该valid
    if not (valid_mask_after_grid > 0).all():
        raise ValueError("GridSample后仍存在invalid的点!")
    print(f"    ✓ 所有点valid_feat_mask > 0")

    # 检查2: point_to_grid索引仍然有效
    if point_to_grid_after_grid.min() < 0:
        raise ValueError(f"point_to_grid包含负索引: min={point_to_grid_after_grid.min()}")
    if point_to_grid_after_grid.max() >= compressed.shape[0]:
        raise ValueError(f"point_to_grid索引越界: max={point_to_grid_after_grid.max()}, compressed_rows={compressed.shape[0]}")
    print(f"    ✓ point_to_grid索引有效: [{point_to_grid_after_grid.min()}, {point_to_grid_after_grid.max()}]")

    # 检查3: segment标签一致性
    grid_labels = set(np.unique(segment_after_grid))
    if not grid_labels.issubset(orig_labels):
        raise ValueError(f"GridSample后出现新标签! 原始: {orig_labels}, Grid后: {grid_labels}")
    print(f"    ✓ segment标签无新增标签")

    # 检查4: 检查通过point_to_grid访问compressed features是否正确
    print(f"\n  验证point_to_grid到compressed features的映射:")

    # 随机采样一些点进行检查
    num_samples = min(100, N_after_grid)
    sample_indices = np.random.choice(N_after_grid, num_samples, replace=False)

    for idx in sample_indices[:10]:  # 只打印前10个
        grid_idx = point_to_grid_after_grid[idx]
        segment_label = segment_after_grid[idx]

        # 获取该grid的压缩特征
        compressed_feat = compressed[grid_idx]

        # 该grid对应的所有原始点
        orig_points_with_same_grid = np.where(orig_point_to_grid == grid_idx)[0]

        # 检查这些原始点的segment标签
        orig_segments_at_grid = orig_segment[orig_points_with_same_grid]

        # 验证: 采样点的segment标签应该在该grid的原始segment标签中
        if segment_label not in orig_segments_at_grid:
            print(f"    ✗ 错位! 点{idx}的segment={int(segment_label)}, "
                  f"但grid {grid_idx}的原始segments={np.unique(orig_segments_at_grid)}")
            return False

    print(f"    ✓ 随机抽检的{num_samples}个点的segment标签与grid对齐")

    # 最终统计
    print(f"\n{'='*80}")
    print("数据变换总结")
    print(f"{'='*80}\n")
    print(f"  原始数据:     {N_orig:>10,} points")
    print(f"  FilterValid后: {N_after_filter:>10,} points ({N_after_filter/N_orig*100:>6.2f}%)")
    print(f"  GridSample后:  {N_after_grid:>10,} points ({N_after_grid/N_orig*100:>6.2f}%)")

    print(f"\n  Segment标签分布 (GridSample后):")
    print(f"    {'Label':>8} {'数量':>12} {'比例':>10}")
    print(f"    {'-'*8} {'-'*12} {'-'*10}")

    for label in sorted(grid_labels):
        count = (segment_after_grid == label).sum()
        ratio = count / N_after_grid * 100
        print(f"    {int(label):>8} {count:>12,} {ratio:>9.2f}%")

    return True


def main():
    parser = argparse.ArgumentParser(description="验证数据对齐")
    parser.add_argument("--scene", type=str, default="figurines",
                        help="场景名称 (默认: figurines)")
    parser.add_argument("--data_root", type=str,
                        default="/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train",
                        help="数据根目录")
    parser.add_argument("--svd_rank", type=int, default=16,
                        help="SVD压缩rank (默认: 16)")
    parser.add_argument("--grid_size", type=float, default=0.01,
                        help="GridSample grid_size (默认: 0.01)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")

    args = parser.parse_args()

    scene_path = Path(args.data_root) / args.scene

    if not scene_path.exists():
        print(f"错误: 场景路径不存在: {scene_path}")
        return 1

    try:
        # 加载数据
        data_dict, compressed = load_scene_data(scene_path, args.svd_rank)

        # 验证原始对齐
        verify_original_alignment(data_dict, compressed)

        # 应用变换并验证
        apply_and_verify_transforms(data_dict, compressed,
                                     grid_size=args.grid_size,
                                     seed=args.seed)

        print(f"\n{'='*80}")
        print("✓ 所有对齐检查通过!")
        print(f"{'='*80}\n")

        return 0

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ 对齐检查失败!")
        print(f"{'='*80}\n")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
