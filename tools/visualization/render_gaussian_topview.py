"""
Render ScanNet++ 3DGS scene (Gaussian point cloud) top-view comparison:
RGB / GT labels / Pred labels, colored with the official ScanNet++ colormap.

No mesh required — renders the Gaussian points directly (coord.npy +
color.npy / segment.npy / pred labels). Reuses get_scannetpp_color_map
from visualize_semantic_segmentation.py.

Usage:
    python tools/visualization/render_gaussian_topview.py \
        --scene /home/isom/cyf/SceneSplat/scannetpp_v2/val/0d2ee665be \
        --pred /path/to/result_ScanNetPPGSDataset/0d2ee665be_pred.npy \
        --output_dir ./output_viz

Outputs (per scene):
    <scene>_rgb_topview.png
    <scene>_gt_topview.png
    <scene>_pred_topview.png
    <scene>_gt.ply  <scene>_pred.ply   (colored point clouds)
"""
import argparse
from pathlib import Path

import numpy as np


def get_scannetpp_color_map():
    """ScanNet++ 100-class colormap (same seeding as visualize_semantic_segmentation)."""
    color_map = {}
    for i in range(100):
        np.random.seed(i * 42 + 123)
        color_map[i] = np.random.randint(50, 256, 3)
    return color_map


def labels_to_colors(labels, color_map, ignore_label=-1):
    """Map label indices to RGB colors (0-255). Unknown/ignore -> light gray."""
    colors = np.full((len(labels), 3), 200, dtype=np.uint8)  # light gray
    for label, color in color_map.items():
        mask = labels == label
        colors[mask] = np.asarray(color)[:3]
    return colors


def render_top_view(coords, colors, output_path, width=1920, height=1080,
                    elev=90, azim=-90, title=None):
    """Top-down (or arbitrary) view render of Gaussian points with matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    step = max(1, len(coords) // 200000)
    c = coords[::step]
    col = colors[::step] / 255.0

    ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=col, s=0.4, alpha=0.8,
               depthshade=False)
    ax.view_init(elev=elev, azim=azim)

    bmin, bmax = c.min(axis=0), c.max(axis=0)
    ax.set_xlim(bmin[0], bmax[0])
    ax.set_ylim(bmin[1], bmax[1])
    ax.set_zlim(bmin[2], bmax[2])
    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    if title:
        ax.set_title(title, fontsize=24)
    plt.savefig(output_path, facecolor="white", dpi=100,
                bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"  ✓ Rendered: {output_path}")


def export_ply(coords, colors, output_path):
    """Export colored point cloud as PLY (readable in MeshLab/CloudCompare)."""
    try:
        import trimesh
        pc = trimesh.PointCloud(vertices=coords, colors=colors)
        pc.export(output_path)
        print(f"  ✓ Exported: {output_path}")
    except ImportError:
        # Fallback: write minimal binary PLY without trimesh
        n = len(coords)
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        body = np.hstack([coords.astype("<f4"), colors.astype("<u1")]).tobytes()
        with open(output_path, "wb") as f:
            f.write(header.encode() + body)
        print(f"  ✓ Exported (trimesh fallback): {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Scene dir (coord.npy etc.)")
    parser.add_argument("--pred", required=True, help="Pred labels npy (full point order)")
    parser.add_argument("--output_dir", default="./output_viz")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--views", default="top", help="top | top,iso")
    args = parser.parse_args()

    scene = Path(args.scene)
    name = scene.name
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    coord = np.load(scene / "coord.npy").astype(np.float32)
    color = np.load(scene / "color.npy").astype(np.uint8)
    seg = np.load(scene / "segment.npy")
    if seg.ndim > 1:
        seg = seg[:, 0]
    pred = np.load(args.pred)
    if pred.ndim > 1:
        pred = pred[:, 0]

    print(f"Scene {name}: {len(coord):,} points | seg {len(seg)} | pred {len(pred)}")
    n = min(len(coord), len(seg), len(pred), len(color))
    if len(seg) != len(pred):
        print(f"  WARNING: seg({len(seg)}) vs pred({len(pred)}) mismatch — truncate to {n}")
    coord, color, seg, pred = coord[:n], color[:n], seg[:n], pred[:n]

    cmap = get_scannetpp_color_map()
    gt_colors = labels_to_colors(seg, cmap)
    pred_colors = labels_to_colors(pred, cmap)

    views = [v.strip() for v in args.views.split(",")]
    for view in views:
        elev, azim = (90, -90) if view == "top" else (30, -60)
        tag = f"_{view}view" if view != "top" else "_topview"
        render_top_view(coord, color, out / f"{name}_rgb{tag}.png",
                        args.width, args.height, elev=elev, azim=azim,
                        title="RGB")
        render_top_view(coord, gt_colors, out / f"{name}_gt{tag}.png",
                        args.width, args.height, elev=elev, azim=azim,
                        title="GT")
        render_top_view(coord, pred_colors, out / f"{name}_pred{tag}.png",
                        args.width, args.height, elev=elev, azim=azim,
                        title="Pred")

    export_ply(coord, gt_colors, out / f"{name}_gt.ply")
    export_ply(coord, pred_colors, out / f"{name}_pred.ply")


if __name__ == "__main__":
    main()
