#!/usr/bin/env python3
"""
Batch Compression Script for Multiple Language Feature Files

This script processes multiple lang_feat files (lang_feat_1.npy, lang_feat_2.npy, lang_feat_3.npy)
by calling compress_grid_svd.py with the appropriate --feat_seq parameter.

Usage:
    # Single scene mode
    python batch_compress_lerf.py --data_dir /path/to/scene --grid_size 0.01

    # Batch mode - all scenes in a dataset
    python batch_compress_lerf.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train_clip/lerf_ovs \\
        --dataset lerf_ovs --split train

    # Specify which feature sequences to process (default: 1,2,3)
    python batch_compress_lerf.py \\
        --data_dir /path/to/scene --feat_seqs 1,2,3

    # All other parameters are passed through to compress_grid_svd.py
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_compression(
    feat_seqs: List[int],
    compress_args: List[str],
) -> None:
    """
    Run compress_grid_svd.py for each feature sequence.

    Args:
        feat_seqs: List of feature sequence numbers to process (e.g., [1, 2, 3])
        compress_args: Arguments to pass to compress_grid_svd.py
    """
    compress_script = PROJECT_ROOT / "tools" / "compress_grid_svd.py"

    print("=" * 70)
    print("Batch Compression for Multiple Language Feature Files")
    print("=" * 70)
    print(f"Feature sequences to process: {feat_seqs}")
    print(f"Arguments passed to compress_grid_svd.py: {' '.join(compress_args)}")
    print("=" * 70)

    success_count = 0
    fail_count = 0
    failed_seqs = []

    for feat_seq in feat_seqs:
        print(f"\n{'=' * 70}")
        print(f"Processing lang_feat_{feat_seq}.npy (Sequence {feat_seq}/{len(feat_seqs)})")
        print(f"{'=' * 70}")

        # Build command
        cmd = [
            sys.executable,
            str(compress_script),
            "--feat_seq", str(feat_seq),
        ] + compress_args

        # Run compress_grid_svd.py
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                success_count += 1
                print(f"  [Success] Sequence {feat_seq} completed successfully")
            else:
                fail_count += 1
                failed_seqs.append(feat_seq)
                print(f"  [Failed] Sequence {feat_seq} failed with return code {result.returncode}")
        except Exception as e:
            fail_count += 1
            failed_seqs.append(feat_seq)
            print(f"  [Error] Exception during sequence {feat_seq}: {e}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("Batch Compression Summary")
    print(f"{'=' * 70}")
    print(f"Total sequences: {len(feat_seqs)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    if failed_seqs:
        print(f"Failed sequences: {failed_seqs}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch compression for multiple language feature files (lang_feat_1.npy, lang_feat_2.npy, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single scene mode - process lang_feat_1, lang_feat_2, lang_feat_3
    python batch_compress_lerf.py --data_dir /path/to/scene --grid_size 0.01

    # Batch mode - all scenes in lerf_ovs/train
    python batch_compress_lerf.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train_clip/lerf_ovs \\
        --dataset lerf_ovs --split train

    # Specify custom feature sequences (e.g., only 1 and 2)
    python batch_compress_lerf.py --data_dir /path/to/scene --feat_seqs 1,2

    # All other parameters are passed through to compress_grid_svd.py
    python batch_compress_lerf.py \\
        --data_dir /path/to/scene \\
        --grid_size 0.01 \\
        --ranks 8,16,32 \\
        --device cuda:0
        """
    )

    # Feature sequence specific arguments
    parser.add_argument(
        "--feat_seqs",
        type=str,
        default="1,2,3",
        help="Comma-separated list of feature sequences to process (default: 1,2,3)",
    )

    # Parse known args to extract batch-specific parameters
    # Remaining args will be passed to compress_grid_svd.py
    args, compress_args = parser.parse_known_args()

    # Parse feature sequences
    feat_seqs = [int(s.strip()) for s in args.feat_seqs.split(',')]

    # Run compression for each feature sequence
    run_compression(feat_seqs, compress_args)


if __name__ == "__main__":
    main()
