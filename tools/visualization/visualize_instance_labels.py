#!/usr/bin/env python
"""
Visualize instance labels from Replica dataset as colored images.

Usage:
    python visualize_instance_labels.py \
        --input /path/to/semantic_instance \
        --output /path/to/output_dir

Reference: visualize_obj function from /new_data/cyf/projects/Gaga/render.py
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import colorsys


def id2rgb(instance_id, max_num_obj=256):
    """
    Convert instance ID to RGB color using golden ratio for color distribution.

    Args:
        instance_id: Instance ID to convert
        max_num_obj: Maximum number of objects for validation

    Returns:
        RGB color as numpy array [3] with values in [0, 255]
    """
    if not 0 <= instance_id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value using golden ratio
    golden_ratio = 1.6180339887
    h = ((instance_id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (instance_id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if instance_id == 0:   # invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb


def visualize_instances(instance_labels):
    """
    Convert instance label image to RGB visualization.

    Args:
        instance_labels: Instance label image (H, W) with integer instance IDs

    Returns:
        RGB visualization image (H, W, 3) with uint8 values
    """
    rgb_mask = np.zeros((*instance_labels.shape[-2:], 3), dtype=np.uint8)
    all_instance_ids = np.unique(instance_labels)

    for instance_id in all_instance_ids:
        colored_mask = id2rgb(instance_id)
        rgb_mask[instance_labels == instance_id] = colored_mask

    return rgb_mask


def main():
    parser = argparse.ArgumentParser(
        description="Visualize instance labels as colored images"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing instance label PNG files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for colored visualization images"
    )
    parser.add_argument(
        "--max-obj",
        type=int,
        default=256,
        help="Maximum number of objects for color generation (default: 256)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to add to output filenames (e.g., '_colored')"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all PNG files from input directory
    input_dir = Path(args.input)
    png_files = sorted(input_dir.glob("*.png"))

    if len(png_files) == 0:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Found {len(png_files)} instance label images")
    print(f"Processing and saving to {output_dir}...")

    # Process each image
    for png_path in tqdm(png_files, desc="Visualizing"):
        # Load instance label image
        instance_img = np.array(Image.open(png_path))

        # Convert to RGB visualization
        rgb_vis = visualize_instances(instance_img)

        # Generate output filename
        stem = png_path.stem
        output_filename = f"{stem}{args.suffix}.png" if args.suffix else f"{stem}.png"
        output_path = output_dir / output_filename

        # Save colored visualization
        Image.fromarray(rgb_vis).save(output_path)

    print(f"Done! Saved {len(png_files)} colored images to {output_dir}")


if __name__ == "__main__":
    main()
