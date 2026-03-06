#!/usr/bin/env python3
"""
Mesh-based Segmentation Visualization Tool

This script visualizes semantic segmentation results on 3D meshes.
It can:
1. Load ScanNet mesh files (.ply)
2. Create sphere meshes from Gaussian splat points
3. Map segmentation predictions to mesh vertices
4. Export colored mesh for visualization

Usage:
    # Visualize as sphere mesh (from Gaussian data)
    python tools/visualize_mesh_segmentation.py \
        --scannetpp_path /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be \
        --mode gt \
        --output output/mesh_gt.ply \
        --render_as_sphere

    # Visualize with predictions
    python tools/visualize_mesh_segmentation.py \
        --scannetpp_path /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be \
        --labels_path /path/to/predictions.npy \
        --mode pred \
        --output output/mesh_pred.ply \
        --render_as_sphere

    # Load existing mesh
    python tools/visualize_mesh_segmentation.py \
        --mesh_path /path/to/mesh.ply \
        --labels_path /path/to/labels.npy \
        --output output/mesh_colored.ply
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import trimesh
from matplotlib.colors import hsv_to_rgb


def get_scannetpp_color_map():
    """ScanNet++ 100-class color map."""
    colors = generate_distinct_colors(100, seed=42)
    return {i: colors[i].tolist() for i in range(100)}


def get_scannet_color_map():
    """ScanNet 20-class color map."""
    colors = {
        0: [0, 0, 0],         # unlabelled/black
        1: [174, 199, 232],   # wall/light blue
        2: [152, 223, 138],   # floor/light green
        3: [31, 119, 180],    # cabinet/blue
        4: [255, 187, 120],   # bed/light orange
        5: [188, 189, 34],    # chair/yellow green
        6: [140, 86, 75],     # table/brown
        7: [255, 152, 150],   # door/light red
        8: [214, 39, 40],     # window/red
        9: [197, 176, 213],   # bookshelf/light purple
        10: [148, 103, 189],  # picture/purple
        11: [196, 156, 148],  # counter/light brown
        12: [23, 190, 207],   # desk/cyan
        13: [247, 182, 210],  # curtain/pink
        14: [219, 219, 141],   # refrigerator/light yellow
        15: [255, 127, 0],    # shower curtain/orange
        16: [158, 218, 229],  # toilet/light cyan
        17: [44, 160, 44],    # sink/green
        18: [112, 128, 144],  # bathtub/gray
        19: [227, 119, 194],  # other furniture/pink
    }
    return colors


def generate_distinct_colors(n=100, seed=42):
    """Generate visually distinct colors using HSV color space."""
    np.random.seed(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    np.random.shuffle(hues)
    saturations = np.random.uniform(0.6, 0.9, n)
    values = np.random.uniform(0.7, 0.95, n)
    hsv_colors = np.stack([hues, saturations, values], axis=1)
    rgb_colors = hsv_to_rgb(hsv_colors)
    return (rgb_colors * 255).astype(np.uint8)


def labels_to_colors(labels, color_map):
    """Convert semantic labels to RGB colors."""
    # Flatten labels if needed (handle [N, 1] arrays)
    labels_flat = labels.flatten() if labels.ndim > 1 else labels

    colors = np.zeros((len(labels_flat), 3), dtype=np.uint8)
    for i, label in enumerate(labels_flat):
        # Convert label to int if it's a numpy type
        label_int = int(label)
        if label_int in color_map:
            colors[i] = color_map[label_int]
        else:
            # Generate random color for unmapped classes
            np.random.seed(label_int % 1000)
            colors[i] = np.random.randint(0, 256, 3)
    return colors


def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    """Load mesh from file."""
    print(f"Loading mesh from: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces: {len(mesh.faces):,}")
    return mesh


def load_ply_with_labels(ply_path: str) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """Load PLY file with vertex labels."""
    import plyfile

    print(f"Loading PLY with labels from: {ply_path}")

    with open(ply_path, 'rb') as f:
        plydata = plyfile.PlyData.read(f)

    # Extract vertices
    vertex = plydata['vertex']
    vertices = np.stack([
        vertex['x'], vertex['y'], vertex['z']
    ], axis=1)

    # Extract labels if available
    if 'label' in vertex.dtype.names:
        labels = vertex['label']
    elif 'category' in vertex.dtype.names:
        labels = vertex['category']
    else:
        labels = np.zeros(len(vertices), dtype=np.int64)

    # Extract faces
    if 'face' in plydata:
        face_data = plydata['face']
        faces = np.stack([face_data['vertex_indices']], axis=1)
        if faces.ndim == 3:
            faces = faces.squeeze(1)
    else:
        # Create faces from vertex count (assuming point cloud)
        faces = np.arange(len(vertices)).reshape(-1, 1)

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    print(f"  Vertices: {len(vertices):,}")
    print(f"  Faces: {len(faces):,}")
    print(f"  Unique labels: {len(np.unique(labels))}")

    return mesh, labels


def map_gaussians_to_mesh(
    gaussian_coords: np.ndarray,
    gaussian_labels: np.ndarray,
    mesh_vertices: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """
    Map Gaussian splat labels to mesh vertices using nearest neighbor.

    Args:
        gaussian_coords: [N, 3] Gaussian coordinates
        gaussian_labels: [N] Gaussian labels
        mesh_vertices: [M, 3] Mesh vertices
        k: Number of nearest neighbors to average over

    Returns:
        [M] Mesh vertex labels
    """
    from sklearn.neighbors import NearestNeighbors

    print(f"Mapping {len(gaussian_coords):,} Gaussians to {len(mesh_vertices):,} vertices...")

    # Build KD-tree on Gaussian coordinates
    nbrs = NearestNeighbors(n_neighbors=min(k, len(gaussian_coords)), algorithm='auto')
    nbrs.fit(gaussian_coords)

    # Find nearest Gaussians for each vertex
    distances, indices = nbrs.kneighbors(mesh_vertices)

    # Assign labels based on nearest neighbors
    if k == 1:
        vertex_labels = gaussian_labels[indices[:, 0]]
    else:
        # Average labels (weighted by inverse distance)
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # For categorical labels, use weighted voting
        vertex_labels = np.zeros(len(mesh_vertices), dtype=np.int64)
        for i in range(len(mesh_vertices)):
            unique_labels, counts = np.unique(gaussian_labels[indices[i]], return_counts=True)
            vertex_labels[i] = unique_labels[np.argmax(counts)]

    print(f"  Mapping complete")
    return vertex_labels


def create_sphere_mesh_from_gaussians(
    coords: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    subdivisions: int = 2,
) -> trimesh.Trimesh:
    """
    Create a mesh by placing spheres at Gaussian locations.

    Args:
        coords: [N, 3] Gaussian centers
        scales: [N, 3] Gaussian scales (for sphere size)
        colors: [N, 3] RGB colors for each sphere
        subdivisions: Sphere subdivision level (higher = more vertices)

    Returns:
        Combined mesh of all spheres
    """
    from trimesh.creation import icosphere

    print(f"Creating sphere mesh from {len(coords):,} Gaussians...")

    # Create base sphere
    base_sphere = icosphere(subdivisions=subdivisions, radius=1.0)

    # Transform and color each sphere
    meshes = []
    vertex_colors = []

    for i in range(len(coords)):
        # Scale sphere by Gaussian scale
        scale = scales[i]
        sphere_scaled = base_sphere.apply_scale(scale)

        # Translate to Gaussian center
        sphere_scaled = sphere_scaled.apply_translation(coords[i])

        meshes.append(sphere_scaled)

        # Add vertex colors for this sphere
        n_vertices = len(sphere_scaled.vertices)
        vertex_colors.extend([colors[i]] * n_vertices)

    # Combine all meshes
    combined_mesh = trimesh.util.concatenate(meshes)

    print(f"  Combined mesh: {len(combined_mesh.vertices):,} vertices, {len(combined_mesh.faces):,} faces")

    return combined_mesh, np.array(vertex_colors, dtype=np.uint8)


def create_point_mesh_with_colors(
    coords: np.ndarray,
    colors: np.ndarray,
    point_size: float = 0.02,
) -> trimesh.Trimesh:
    """
    Create a simple point mesh using small spheres or colored points.

    Args:
        coords: [N, 3] point coordinates
        colors: [N, 3] RGB colors in [0, 255] range
        point_size: Size of point markers

    Returns:
        Mesh with colored vertices
    """
    print(f"Creating point mesh from {len(coords):,} points...")

    # For large point clouds, use point rendering (faces are single vertices)
    # This creates a point cloud that can be saved as PLY
    from trimesh.creation import icosphere

    # Create a small sphere for each point (use low subdivision for efficiency)
    if len(coords) > 100000:
        # Too many points - use point cloud instead
        print("  Warning: Large point cloud, using point rendering")
        faces = np.arange(len(coords)).reshape(-1, 1)
        mesh = trimesh.Trimesh(vertices=coords, faces=faces)
        return mesh, colors

    # Create small spheres
    base_sphere = icosphere(subdivisions=1, radius=point_size)
    meshes = []
    vertex_colors = []

    for i in range(len(coords)):
        sphere = base_sphere.apply_translation(coords[i])
        meshes.append(sphere)
        vertex_colors.extend([colors[i]] * len(sphere.vertices))

    combined_mesh = trimesh.util.concatenate(meshes)

    print(f"  Created mesh: {len(combined_mesh.vertices):,} vertices")

    return combined_mesh, np.array(vertex_colors, dtype=np.uint8)


def find_scannet_mesh(scannetpp_scene_id: str, scannet_base_path: str) -> Optional[str]:
    """
    Find ScanNet mesh file corresponding to ScanNet++ scene.

    Args:
        scannetpp_scene_id: ScanNet++ scene ID (e.g., '0d2ee665be')
        scannet_base_path: Base path to ScanNet scans directory

    Returns:
        Path to ScanNet mesh file, or None if not found
    """
    # Try common mappings - this would need a proper mapping file in production
    # For now, we'll try some heuristics

    scannet_base = Path(scannet_base_path)

    # Option 1: Check if there's a mapping file
    mapping_files = [
        Path("/new_data/cyf/projects/SceneSplat/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/scans.txt"),
        Path("/new_data/cyf/Datasets/SceneSplat7k/data_splits/scannetpp_v2/scans_mapping.txt"),
    ]

    for mapping_file in mapping_files:
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                for line in f:
                    if scannetpp_scene_id in line:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            scannet_id = parts[1] if parts[1] != scannetpp_scene_id else parts[0]
                            mesh_path = scannet_base / scannet_id / f"{scannet_id}_vh_clean_2.ply"
                            if mesh_path.exists():
                                return str(mesh_path)

    # Option 2: Direct search (might not work due to different ID formats)
    # ScanNet++ uses hash IDs, ScanNet uses scene####_## format

    return None


def export_colored_mesh(
    mesh: trimesh.Trimesh,
    colors: np.ndarray,
    output_path: str,
) -> None:
    """
    Export mesh with vertex colors.

    Args:
        mesh: Trimesh object
        colors: [N, 3] RGB colors in [0, 255] range
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Create a copy with vertex colors
    mesh_colored = mesh.copy()

    # Add vertex colors to the mesh
    mesh_colored.visual = trimesh.visual.ColorVisuals(
        vertex_colors=colors.astype(np.uint8)
    )

    # Export
    mesh_colored.export(output_path)
    print(f"Exported colored mesh to: {output_path}")


def export_ply_with_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    output_path: str,
) -> None:
    """
    Export mesh to PLY format with vertex colors.

    Args:
        vertices: [N, 3] vertex coordinates
        faces: [M, 3] face indices
        colors: [N, 3] RGB colors in [0, 255] range
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_vertices = len(vertices)
    n_faces = len(faces)

    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")

        # Write vertices with colors
        for i in range(n_vertices):
            f.write(f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f} ")
            f.write(f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")

        # Write faces
        for i in range(n_faces):
            f.write(f"3 {faces[i, 0]} {faces[i, 1]} {faces[i, 2]}\n")

    print(f"Exported PLY to: {output_path}")


def render_mesh_to_image(
    mesh_path: str,
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """
    Render mesh to 2D image using matplotlib.

    Args:
        mesh_path: Path to colored mesh file
        output_path: Output image path
        width: Image width
        height: Image height
        background_color: Background RGB color
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Load mesh
    mesh = load_mesh(mesh_path)

    # Setup figure
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.set_axis_off()

    # Set background color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Get vertex colors if available
    if hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors
    else:
        colors = None

    # Plot mesh
    mesh.plot(ax, color=colors)

    # Set view
    ax.view_init(elev=30, azim=45)

    # Render
    fig.savefig(output_path, dpi=dpi, facecolor=background_color, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Rendered to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation on 3D meshes")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--mesh_path", type=str,
                            help="Path to mesh file (.ply)")
    input_group.add_argument("--scannetpp_path", type=str,
                            help="Path to ScanNet++ scene directory")

    # Labels
    parser.add_argument("--labels_path", type=str,
                        help="Path to labels/predictions file (.npy)")
    parser.add_argument("--scannet_path", type=str,
                        default="/new_data/cyf/Datasets/ScanNet/scans",
                        help="Path to ScanNet scans directory (for finding original mesh)")

    # Mode
    parser.add_argument("--mode", type=str, choices=["gt", "pred"], default="gt",
                        help="Visualization mode")
    parser.add_argument("--dataset", type=str, choices=["scannet", "scannetpp"], default="scannetpp",
                        help="Dataset type for color mapping")

    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for colored mesh")
    parser.add_argument("--render_image", action="store_true",
                        help="Also render to 2D image")
    parser.add_argument("--render_width", type=int, default=1920,
                        help="Render image width")
    parser.add_argument("--render_height", type=int, default=1080,
                        help="Render image height")

    # Mesh creation options
    parser.add_argument("--render_as_sphere", action="store_true",
                        help="Render Gaussians as spheres (uses scale data)")
    parser.add_argument("--sphere_subdivisions", type=int, default=2,
                        help="Sphere subdivision level (1=faster, 3=smoother)")
    parser.add_argument("--point_size", type=float, default=0.02,
                        help="Size of point markers when not using spheres")
    parser.add_argument("--max_points", type=int, default=100000,
                        help="Maximum number of points to render (for large scenes, will downsample)")

    # Mapping options
    parser.add_argument("--knn", type=int, default=1,
                        help="Number of nearest neighbors for label mapping")

    args = parser.parse_args()

    # Get color map
    if args.dataset == "scannet":
        color_map = get_scannet_color_map()
    else:
        color_map = get_scannetpp_color_map()

    # Load mesh
    if args.mesh_path:
        # Direct mesh file
        if args.labels_path is None:
            # Check if mesh has labels
            try:
                mesh, labels = load_ply_with_labels(args.mesh_path)
            except:
                raise ValueError("Labels must be provided via --labels_path for this mesh")
        else:
            mesh = load_mesh(args.mesh_path)
            labels = np.load(args.labels_path)
    else:
        # ScanNet++ directory - need to find corresponding ScanNet mesh
        scannetpp_path = Path(args.scannetpp_path)
        scene_id = scannetpp_path.name

        # Check for ScanNet mesh mapping
        scannet_mesh = find_scannet_mesh(scene_id, args.scannet_path)

        if scannet_mesh:
            print(f"Found ScanNet mesh: {scannet_mesh}")
            mesh, gt_labels = load_ply_with_labels(scannet_mesh)
        else:
            print(f"Warning: Could not find ScanNet mesh for scene {scene_id}")
            print("Creating mesh from Gaussian data...")

            # Load Gaussian/point cloud data
            coord_path = scannetpp_path / "coord.npy"
            segment_path = scannetpp_path / "segment.npy"
            scale_path = scannetpp_path / "scale.npy"
            color_path = scannetpp_path / "color.npy"

            if not coord_path.exists():
                raise ValueError(f"Cannot find coordinates in {scannetpp_path}")

            coords = np.load(coord_path)
            scales = np.load(scale_path) if scale_path.exists() else np.ones((len(coords), 3)) * 0.02

            if segment_path.exists():
                gt_labels = np.load(segment_path)
            elif args.labels_path:
                gt_labels = np.load(args.labels_path)
            else:
                raise ValueError("No labels found")

            # Use appropriate labels based on mode
            if args.mode == "gt":
                labels = gt_labels
            elif args.labels_path:
                labels = np.load(args.labels_path)
            else:
                labels = gt_labels

            # Downsample if too many points
            num_points = len(coords)
            if num_points > args.max_points:
                print(f"Downsampling from {num_points:,} to {args.max_points:,} points...")
                indices = np.random.choice(num_points, args.max_points, replace=False)
                coords = coords[indices]
                scales = scales[indices]
                labels = labels[indices]
                print(f"  Downsampled to {len(coords):,} points")

            # Convert labels to colors
            colors_rgb = labels_to_colors(labels, color_map)

            # Disable sphere rendering for large point clouds (too slow)
            if args.render_as_sphere and len(coords) > 50000:
                print(f"Warning: Sphere rendering disabled for {len(coords):,} points (too slow)")
                print(f"         Using point cloud rendering instead")
                args.render_as_sphere = False

            # Create mesh from Gaussians
            if args.render_as_sphere:
                mesh, vertex_colors = create_sphere_mesh_from_gaussians(
                    coords, scales, colors_rgb,
                    subdivisions=args.sphere_subdivisions
                )
                colors = vertex_colors
            else:
                # Create point cloud mesh
                mesh, colors = create_point_mesh_with_colors(
                    coords, colors_rgb, point_size=args.point_size
                )

            # Export colored mesh
            if hasattr(mesh, 'visual'):
                mesh.visual.vertex_colors = colors

            export_ply_with_colors(mesh.vertices, mesh.faces, colors, args.output)

            if args.render_image:
                image_path = args.output.replace('.ply', '_render.png')
                render_mesh_to_image(args.output, image_path, args.render_width, args.render_height)

            return

        # Use appropriate labels based on mode
        if args.mode == "gt":
            labels = gt_labels
        elif args.labels_path:
            labels = np.load(args.labels_path)
            # May need to map to mesh vertices
            if len(labels) != len(mesh.vertices):
                print(f"Label count mismatch: {len(labels)} vs {len(mesh.vertices)} vertices")
                print("Attempting Gaussian-to-mesh mapping...")
                # This would require Gaussian coordinates
                labels = map_gaussians_to_mesh(
                    np.load(scannetpp_path / "coord.npy"),
                    labels,
                    mesh.vertices,
                    k=args.knn,
                )

    # Convert labels to colors
    colors = labels_to_colors(labels, color_map)

    # Export colored mesh
    export_ply_with_colors(mesh.vertices, mesh.faces, colors, args.output)

    # Optionally render to image
    if args.render_image:
        image_path = args.output.replace('.ply', '_render.png')
        render_mesh_to_image(args.output, image_path, args.render_width, args.render_height)


if __name__ == "__main__":
    main()
