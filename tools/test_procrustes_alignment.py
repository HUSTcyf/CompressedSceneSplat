#!/usr/bin/env python3
"""
Test Script for Orthogonal Procrustes Q Matrix Computation

This script tests the Procrustes alignment functionality with synthetic data
to verify correctness of the implementation.

Usage:
    python tools/test_procrustes_alignment.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_synthetic_data():
    """Test Procrustes alignment with synthetic data."""
    from tools.compute_procrustes_alignment import (
        find_unique_rows,
        find_row_correspondence,
        perform_svd_reduction,
        compute_procrustes_Q,
    )

    print("="*70)
    print("Test 1: Synthetic Data - Alignment Improves Similarity")
    print("="*70)

    # Parameters
    N = 1000  # Original points
    M = 200   # Unique points
    D = 50    # Original dimension
    d = 16    # Target dimension

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random high-dimensional data A
    A = np.random.randn(N, D).astype(np.float32)

    # Select M unique rows from A to form B
    unique_indices = np.random.choice(N, M, replace=False)
    unique_indices.sort()
    B = A[unique_indices]

    print(f"  Data shapes:")
    print(f"    A: {A.shape}")
    print(f"    B: {B.shape}")

    # Step 1: SVD reduction
    print(f"\n  Step 1: SVD reduction to {d} dimensions...")
    X, V_A = perform_svd_reduction(A, d, normalize=True)
    Y, V_B = perform_svd_reduction(B, d, normalize=True)

    print(f"    X shape: {X.shape}")
    print(f"    Y shape: {Y.shape}")

    # Step 2: Extract corresponding rows
    print(f"\n  Step 2: Extract corresponding rows from X...")
    X_c = X[unique_indices]
    print(f"    X_c shape: {X_c.shape}")

    # Step 3: Apply random rotation to X_c to simulate misalignment
    print(f"\n  Step 3: Apply random rotation to X_c (simulate misalignment)...")
    Q_random = np.random.randn(d, d)
    Q_rot, _ = np.linalg.qr(Q_random)
    X_c_rotated = X_c @ Q_rot

    # Compute cosine similarity before alignment
    cos_before = np.mean([
        np.dot(X_c_rotated[i], Y[i]) / (np.linalg.norm(X_c_rotated[i]) * np.linalg.norm(Y[i]) + 1e-8)
        for i in range(Y.shape[0])
    ])
    print(f"    Cosine similarity (before alignment): {cos_before:.4f}")

    # Step 4: Compute Procrustes Q to align
    print(f"\n  Step 4: Compute Procrustes Q for alignment...")
    Q_est, metrics = compute_procrustes_Q(X_c_rotated, Y, d)

    # Compute orthogonality error
    QtQ = Q_est.T @ Q_est
    orthogonality_error = np.linalg.norm(QtQ - np.eye(d), 'fro')

    print(f"\n  Results:")
    print(f"    Estimated Q shape: {Q_est.shape}")
    print(f"    Orthogonality error: {orthogonality_error:.8e}")
    print(f"    Alignment residual: {metrics['residual_norm']:.6f}")
    print(f"    Relative error: {metrics['relative_error']:.6f}")
    print(f"    Cosine similarity (after): {metrics['cosine_after']:.4f}")

    # Test passed if:
    # 1. Q is orthogonal
    # 2. Alignment improves cosine similarity
    # 3. det(Q) is close to 1
    test_passed = (
        orthogonality_error < 1e-5 and
        metrics['cosine_after'] > cos_before and
        abs(np.linalg.det(Q_est) - 1.0) < 1e-5
    )

    if test_passed:
        print(f"\n  ✓ Test PASSED: Q matrix is orthogonal and alignment improves similarity!")
    else:
        print(f"\n  ✗ Test FAILED")

    return test_passed


def test_unique_row_finding():
    """Test unique row finding functionality."""
    from tools.compute_procrustes_alignment import find_unique_rows, find_row_correspondence

    print("\n" + "="*70)
    print("Test 2: Unique Row Finding")
    print("="*70)

    # Create data with duplicate rows
    np.random.seed(42)
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [1.0, 2.0, 3.0],  # Duplicate of row 0
        [7.0, 8.0, 9.0],
        [4.0, 5.0, 6.0],  # Duplicate of row 1
    ], dtype=np.float32)

    # Find unique rows
    unique_arr, inverse = find_unique_rows(data)

    print(f"  Original data shape: {data.shape}")
    print(f"  Unique rows shape: {unique_arr.shape}")
    print(f"  Inverse mapping shape: {inverse.shape}")

    # Verify unique rows
    expected_unique = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float32)

    if np.allclose(unique_arr, expected_unique):
        print(f"  ✓ Unique rows found correctly")
    else:
        print(f"  ✗ Unique rows incorrect")
        return False

    # Verify inverse mapping
    # inverse[i] should give the index of the unique row for row i
    expected_inverse = np.array([0, 1, 0, 2, 1])
    if np.array_equal(inverse, expected_inverse):
        print(f"  ✓ Inverse mapping correct")
    else:
        print(f"  ✗ Inverse mapping incorrect")
        print(f"    Expected: {expected_inverse}")
        print(f"    Got: {inverse}")
        return False

    # Test correspondence finding
    # Correspondence maps each unique row to its FIRST occurrence in original array
    # unique[0] = [1,2,3] -> first at index 0
    # unique[1] = [4,5,6] -> first at index 1
    # unique[2] = [7,8,9] -> first at index 3
    correspondence = find_row_correspondence(data, unique_arr)
    expected_corr = np.array([0, 1, 3])  # First occurrence indices for each unique row
    if np.array_equal(correspondence, expected_corr):
        print(f"  ✓ Correspondence found correctly")
    else:
        print(f"  ✗ Correspondence incorrect")
        print(f"    Expected: {expected_corr}")
        print(f"    Got: {correspondence}")
        return False

    return True


def test_orthogonality():
    """Test that computed Q matrices are orthogonal."""
    from tools.compute_procrustes_alignment import compute_procrustes_Q

    print("\n" + "="*70)
    print("Test 3: Orthogonality Constraint")
    print("="*70)

    np.random.seed(42)
    N, d = 100, 16

    # Generate random matrices
    X_c = np.random.randn(N, d).astype(np.float32)
    Y = np.random.randn(N, d).astype(np.float32)

    # Compute Q
    Q, metrics = compute_procrustes_Q(X_c, Y, d)

    # Check orthogonality: Q^T @ Q should be identity
    QtQ = Q.T @ Q
    identity_error = np.linalg.norm(QtQ - np.eye(d), 'fro')

    print(f"  Q shape: {Q.shape}")
    print(f"  ||Q^T @ Q - I||_F: {identity_error:.8e}")
    print(f"  det(Q): {np.linalg.det(Q):.6f}")

    # Check if det(Q) is close to 1 (proper rotation, not reflection)
    det_close_to_1 = abs(np.linalg.det(Q) - 1.0) < 1e-5

    if identity_error < 1e-5 and det_close_to_1:
        print(f"  ✓ Q is orthogonal with det(Q) ≈ 1")
        return True
    else:
        print(f"  ✗ Q is not properly orthogonal")
        return False


def test_alignment_quality():
    """Test that alignment improves similarity."""
    from tools.compute_procrustes_alignment import perform_svd_reduction, compute_procrustes_Q

    print("\n" + "="*70)
    print("Test 4: Alignment Quality Improvement")
    print("="*70)

    np.random.seed(42)
    N, D, d = 500, 100, 16

    # Generate data
    A = np.random.randn(N, D).astype(np.float32)
    B = A[np.random.choice(N, 100, replace=False)]

    # SVD reduction
    X, _ = perform_svd_reduction(A, d, normalize=True)
    Y, _ = perform_svd_reduction(B, d, normalize=True)

    # Create correspondence
    correspondence = np.random.choice(X.shape[0], Y.shape[0], replace=False)
    X_c = X[correspondence]

    # Apply random rotation to X_c
    Q_random = np.random.randn(d, d)
    Q_rot, _ = np.linalg.qr(Q_random)
    X_c_rotated = X_c @ Q_rot

    # Compute cosine similarity before alignment
    cos_before = np.mean([
        np.dot(X_c_rotated[i], Y[i]) / (np.linalg.norm(X_c_rotated[i]) * np.linalg.norm(Y[i]) + 1e-8)
        for i in range(Y.shape[0])
    ])

    # Compute Q and align
    Q, metrics = compute_procrustes_Q(X_c_rotated, Y, d)
    X_aligned = X_c_rotated @ Q

    # Compute cosine similarity after alignment
    cos_after = np.mean([
        np.dot(X_aligned[i], Y[i]) / (np.linalg.norm(X_aligned[i]) * np.linalg.norm(Y[i]) + 1e-8)
        for i in range(Y.shape[0])
    ])

    print(f"  Cosine similarity (before): {cos_before:.4f}")
    print(f"  Cosine similarity (after): {cos_after:.4f}")
    print(f"  Improvement: {cos_after - cos_before:+.4f}")

    if cos_after > cos_before:
        print(f"  ✓ Alignment improved similarity")
        return True
    else:
        print(f"  ✗ Alignment did not improve similarity")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Orthogonal Procrustes Q Matrix Computation Tests")
    print("="*70)

    results = []

    # Run tests
    results.append(("Synthetic Data Test", test_synthetic_data()))
    results.append(("Unique Row Finding", test_unique_row_finding()))
    results.append(("Orthogonality Constraint", test_orthogonality()))
    results.append(("Alignment Quality", test_alignment_quality()))

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n❌ {total_tests - total_passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
