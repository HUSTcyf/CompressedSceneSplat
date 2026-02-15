#!/usr/bin/env python3
"""
Check COLMAP dataset for common issues.
Usage: python check_colmap_dataset.py <data_dir>
"""
import os
import sys
import imageio.v2 as imageio
import numpy as np
from pathlib import Path

def check_colmap_dataset(data_dir):
    """Check COLMAP dataset for common issues."""
    print(f"Checking COLMAP dataset: {data_dir}")
    print("=" * 60)

    issues = []

    # Check sparse directory
    sparse_dir = Path(data_dir) / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = Path(data_dir) / "sparse"
    if not sparse_dir.exists():
        issues.append(f"Sparse directory not found: {sparse_dir}")
    else:
        print(f"✓ Sparse directory found: {sparse_dir}")

    # Check for required COLMAP files
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    for fname in required_files:
        fpath = sparse_dir / fname
        if not fpath.exists():
            issues.append(f"Required file not found: {fpath}")
        else:
            print(f"✓ Found: {fname}")

    # Try to load COLMAP data
    try:
        from pycolmap import SceneManager
        manager = SceneManager(str(sparse_dir))
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()
        print(f"✓ COLMAP data loaded successfully")
        print(f"  - {len(manager.cameras)} cameras")
        print(f"  - {len(manager.images)} images")
        print(f"  - {len(manager.points3D)} 3D points")
    except Exception as e:
        issues.append(f"Failed to load COLMAP data: {e}")

    # Check images directory
    images_dir = Path(data_dir) / "images"
    if not images_dir.exists():
        issues.append(f"Images directory not found: {images_dir}")
    else:
        print(f"✓ Images directory found: {images_dir}")

        # Check image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if len(image_files) == 0:
            issues.append("No image files found in images directory")
        else:
            print(f"✓ Found {len(image_files)} image files")

            # Sample check a few images
            print("\n  Checking sample images...")
            for i, img_path in enumerate(image_files[:5]):
                try:
                    img = imageio.imread(img_path)
                    print(f"  ✓ {img_path.name}: {img.shape}")
                except Exception as e:
                    issues.append(f"Failed to load image {img_path.name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  ✗ {issue}")
        return 1
    else:
        print("✓ No issues found! Dataset looks good.")
        return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_colmap_dataset.py <data_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]
    sys.exit(check_colmap_dataset(data_dir))
