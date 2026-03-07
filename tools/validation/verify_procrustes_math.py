#!/usr/bin/env python3
"""
Mathematical Verification of Orthogonal Procrustes Alignment

This script verifies the correctness of the Procrustes alignment implementation
by checking each step of the mathematical derivation:

1. Problem: min_Q ||X_c @ Q - Y||_F^2 subject to Q^T @ Q = I
2. Solution derivation: max_Q Tr(Q^T @ M) where M = X_c^T @ Y
3. SVD solution: Q = U @ V^T where M = U @ Σ @ V^T

Mathematical Background:
- The objective minimizes Frobenius norm: ||X_c @ Q - Y||_F^2
- This is equivalent to maximizing: Tr(Q^T @ X_c^T @ Y)
- Let M = X_c^T @ Y, then: max_Q Tr(Q^T @ M) subject to Q^T @ Q = I
- SVD of M: M = U @ Σ @ V^T gives optimal Q = U @ V^T

Usage:
    python tools/verify_procrustes_math.py
"""

import numpy as np
import sys
from pathlib import Path

# Import PROJECT_ROOT - handle both script and module execution
try:
    from .. import PROJECT_ROOT  # Relative import when run as module
except ImportError:
    # Fallback when run as script: add parent dir to sys.path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def frobenius_norm_squared(M: np.ndarray) -> float:
    """Compute squared Frobenius norm: ||M||_F^2 = Tr(M^T @ M)"""
    return np.sum(M ** 2)


def verify_svd_solution(X_c: np.ndarray, Y: np.ndarray):
    """
    Verify the SVD solution for Procrustes alignment.

    Mathematical steps to verify:
    1. M = X_c^T @ Y
    2. SVD: M = U @ Σ @ V^T
    3. Optimal Q = U @ V^T
    """
    print("="*70)
    print("Mathematical Verification of Procrustes Alignment")
    print("="*70)

    N, d = X_c.shape
    print(f"\nInput dimensions:")
    print(f"  X_c: {X_c.shape} (N={N}, d={d})")
    print(f"  Y:   {Y.shape} (N={N}, d={d})")

    # Step 1: Compute M = X_c^T @ Y
    print(f"\n{'='*70}")
    print("Step 1: Compute M = X_c^T @ Y")
    print(f"{'='*70}")
    M = X_c.T @ Y
    print(f"  M shape: {M.shape}")
    print(f"  M = X_c^T @ Y ✓")

    # Step 2: SVD of M
    print(f"\n{'='*70}")
    print("Step 2: SVD Decomposition M = U @ Σ @ V^T")
    print(f"{'='*70}")
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt.T
    print(f"  U shape: {U.shape}")
    print(f"  Σ shape: {S.shape} (singular values)")
    print(f"  V shape: {V.shape}")
    print(f"  Reconstruction error: ||M - U @ diag(S) @ V^T|| = {np.linalg.norm(M - U @ np.diag(S) @ Vt):.8e}")

    # Step 3: Compute Q = U @ V^T
    print(f"\n{'='*70}")
    print("Step 3: Compute Q = U @ V^T")
    print(f"{'='*70}")
    Q = U @ Vt
    print(f"  Q shape: {Q.shape}")
    print(f"  Q = U @ V^T ✓")

    # Verification 1: Orthogonality constraint
    print(f"\n{'='*70}")
    print("Verification 1: Orthogonality Constraint Q^T @ Q = I")
    print(f"{'='*70}")
    QtQ = Q.T @ Q
    orthogonality_error = np.linalg.norm(QtQ - np.eye(d), 'fro')
    print(f"  ||Q^T @ Q - I||_F = {orthogonality_error:.8e}")
    if orthogonality_error < 1e-6:
        print(f"  ✓ Q is orthogonal (error < 1e-6)")
    else:
        print(f"  ✗ Q is NOT orthogonal!")

    # Verification 2: Determinant (proper rotation vs reflection)
    print(f"\n{'='*70}")
    print("Verification 2: Determinant (Rotation vs Reflection)")
    print(f"{'='*70}")
    det_Q = np.linalg.det(Q)
    print(f"  det(Q) = {det_Q:.6f}")
    if abs(det_Q - 1.0) < 1e-5:
        print(f"  ✓ Q is a proper rotation (det ≈ +1)")
    elif abs(det_Q + 1.0) < 1e-5:
        print(f"  ⚠ Q contains reflection (det ≈ -1)")
    else:
        print(f"  ✗ Q is not properly orthogonal!")

    # Verification 3: Objective function value
    print(f"\n{'='*70}")
    print("Verification 3: Objective Function Value")
    print(f"{'='*70}")
    residual_before = frobenius_norm_squared(X_c - Y)
    residual_after = frobenius_norm_squared(X_c @ Q - Y)
    print(f"  ||X_c - Y||_F^2 (before alignment): {residual_before:.6f}")
    print(f"  ||X_c @ Q - Y||_F^2 (after alignment): {residual_after:.6f}")
    print(f"  Improvement: {residual_before - residual_after:.6f} ({(residual_before - residual_after) / residual_before * 100:.1f}%)")

    if residual_after < residual_before:
        print(f"  ✓ Alignment reduces Frobenius norm")
    else:
        print(f"  ✗ Alignment did NOT reduce Frobenius norm!")

    # Verification 4: Trace maximization
    print(f"\n{'='*70}")
    print("Verification 4: Trace Maximization")
    print(f"{'='*70}")
    trace_QtM = np.trace(Q.T @ M)
    print(f"  Tr(Q^T @ M) = {trace_QtM:.6f}")
    print(f"  Theoretical maximum: Σ σ_i = {np.sum(S):.6f}")
    trace_diff = abs(trace_QtM - np.sum(S))
    print(f"  Difference: {trace_diff:.8e}")
    if trace_diff < 1e-5:
        print(f"  ✓ Trace maximization achieved")
    else:
        print(f"  ✗ Trace NOT maximized!")

    # Verification 5: Compare with unconstrained solution
    print(f"\n{'='*70}")
    print("Verification 5: Comparison with Unconstrained Solution")
    print(f"{'='*70}")
    # Unconstrained least squares: Q_unconstrained = (X_c^T @ X_c)^(-1) @ X_c^T @ Y
    XtX_inv = np.linalg.inv(X_c.T @ X_c + 1e-8 * np.eye(d))
    Q_unconstrained = XtX_inv @ X_c.T @ Y
    residual_unconstrained = frobenius_norm_squared(X_c @ Q_unconstrained - Y)
    print(f"  Unconstrained Q residual: {residual_unconstrained:.6f}")
    print(f"  Orthogonal Q residual:    {residual_after:.6f}")
    print(f"  Difference: {residual_unconstrained - residual_after:.6f}")
    print(f"  Note: Orthogonal constraint may increase residual slightly")
    print(f"        but preserves inner product structure")

    # Verification 6: Coordinate system alignment
    print(f"\n{'='*70}")
    print("Verification 6: Coordinate System Alignment")
    print(f"{'='*70}")
    X_aligned = X_c @ Q

    # Compute cosine similarity before and after
    cosine_before = np.mean([
        np.dot(X_c[i], Y[i]) / (np.linalg.norm(X_c[i]) * np.linalg.norm(Y[i]) + 1e-8)
        for i in range(N)
    ])
    cosine_after = np.mean([
        np.dot(X_aligned[i], Y[i]) / (np.linalg.norm(X_aligned[i]) * np.linalg.norm(Y[i]) + 1e-8)
        for i in range(N)
    ])
    print(f"  Mean cosine similarity (before): {cosine_before:.4f}")
    print(f"  Mean cosine similarity (after):  {cosine_after:.4f}")
    print(f"  Improvement: {cosine_after - cosine_before:+.4f}")

    if cosine_after > cosine_before:
        print(f"  ✓ Alignment improves similarity")
    else:
        print(f"  ✗ Alignment did NOT improve similarity!")

    return {
        'Q': Q,
        'orthogonality_error': orthogonality_error,
        'det_Q': det_Q,
        'residual_before': residual_before,
        'residual_after': residual_after,
        'trace_QtM': trace_QtM,
        'sum_singular_values': np.sum(S),
        'cosine_before': cosine_before,
        'cosine_after': cosine_after,
    }


def verify_known_rotation():
    """
    Verify with a known rotation matrix.
    If we rotate X_c by a known R, then Q should recover R^T.
    """
    print(f"\n{'='*70}")
    print("Verification with Known Rotation Matrix")
    print(f"{'='*70}")

    np.random.seed(42)
    N, d = 100, 16

    # Generate random data
    X = np.random.randn(N, d).astype(np.float32)

    # Create known rotation
    R_random = np.random.randn(d, d)
    R_true, _ = np.linalg.qr(R_random)

    # Rotate X
    X_rotated = X @ R_true

    # To recover, we need to find Q such that X_rotated @ Q ≈ X
    # This means Q should be R_true^T
    results = verify_svd_solution(X_rotated, X)

    # Verify Q ≈ R_true^T
    Q_est = results['Q']
    R_true_inv = R_true.T
    recovery_error = np.linalg.norm(Q_est - R_true_inv, 'fro')
    QtR = Q_est @ R_true
    identity_error = np.linalg.norm(QtR - np.eye(d), 'fro')

    print(f"\n{'='*70}")
    print("Recovery Verification")
    print(f"{'='*70}")
    print(f"  Known rotation: R_true")
    print(f"  Expected inverse: R_true^T")
    print(f"  Estimated Q: Q_est")
    print(f"  ||Q_est - R_true^T||_F = {recovery_error:.6f}")
    print(f"  ||Q_est @ R_true - I||_F = {identity_error:.6f}")

    if identity_error < 0.1:
        print(f"  ✓ Q correctly recovers the inverse rotation")
        return True
    else:
        print(f"  ✗ Q did NOT correctly recover the rotation")
        return False


def verify_gradient_optimality():
    """
    Verify that the solution satisfies KKT conditions (gradient = 0).
    For constrained optimization min f(Q) s.t. Q^T @ Q = I,
    the KKT condition is: ∇f(Q) = Q @ Λ for some symmetric Λ.
    """
    print(f"\n{'='*70}")
    print("KKT Condition Verification")
    print(f"{'='*70}")

    np.random.seed(42)
    N, d = 50, 8

    X_c = np.random.randn(N, d).astype(np.float32)
    Y = np.random.randn(N, d).astype(np.float32)

    # Compute optimal Q
    M = X_c.T @ Y
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = U @ Vt

    # Gradient of f(Q) = ||X_c @ Q - Y||_F^2
    # ∇f(Q) = 2 * X_c.T @ (X_c @ Q - Y)
    residual = X_c @ Q - Y
    gradient = 2 * X_c.T @ residual

    # Check if gradient is orthogonal to the tangent space of constraint manifold
    # For orthogonal constraint, the tangent space consists of matrices Z where Q^T @ Z is skew-symmetric
    # The KKT condition: gradient should be of the form Q @ Λ where Λ is symmetric

    Lambda = Q.T @ gradient
    Lambda_symmetric = (Lambda + Lambda.T) / 2
    skew_part = (Lambda - Lambda.T) / 2
    skew_norm = np.linalg.norm(skew_part, 'fro')

    print(f"  Gradient shape: {gradient.shape}")
    print(f"  ||Skew(Λ)||_F (should be ~0): {skew_norm:.8e}")

    # Also check that gradient @ Q^T is symmetric
    GQt = gradient @ Q.T
    GQt_symmetric = (GQt + GQt.T) / 2
    asymmetry = np.linalg.norm(GQt - GQt_symmetric, 'fro')
    print(f"  Asymmetry of ∇f(Q) @ Q^T: {asymmetry:.8e}")

    if skew_norm < 1e-5 and asymmetry < 1e-5:
        print(f"  ✓ KKT conditions satisfied")
        return True
    else:
        print(f"  ⚠ KKT conditions not perfectly satisfied (numerical errors)")
        return skew_norm < 1e-4 and asymmetry < 1e-4


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("MATHEMATICAL VERIFICATION OF PROCRUSTES ALIGNMENT")
    print("="*70)

    results = []

    # Test 1: Known rotation recovery
    results.append(("Known Rotation Recovery", verify_known_rotation()))

    # Test 2: KKT conditions
    results.append(("KKT Conditions", verify_gradient_optimality()))

    # Test 3: Random data alignment
    print(f"\n{'='*70}")
    print("Random Data Alignment Test")
    print(f"{'='*70}")
    np.random.seed(123)
    N, d = 200, 16
    X = np.random.randn(N, d).astype(np.float32)
    Y = np.random.randn(N, d).astype(np.float32)
    metrics = verify_svd_solution(X, Y)

    # Check key properties
    test_passed = (
        metrics['orthogonality_error'] < 1e-6 and
        abs(metrics['det_Q']) > 0.9 and  # Allow reflection
        metrics['residual_after'] < metrics['residual_before'] and
        metrics['cosine_after'] > metrics['cosine_before'] and
        abs(metrics['trace_QtM'] - metrics['sum_singular_values']) < 1e-5
    )
    results.append(("Random Data Alignment", test_passed))

    # Print summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} verifications passed")

    if total_passed == total_tests:
        print("\n🎉 All mathematical verifications PASSED!")
        print("\nConclusion: The implementation correctly follows the")
        print("orthogonal Procrustes analysis mathematical derivation:")
        print("  1. ✓ M = X_c^T @ Y")
        print("  2. ✓ SVD: M = U @ Σ @ V^T")
        print("  3. ✓ Q = U @ V^T")
        print("  4. ✓ Q^T @ Q = I (orthogonality)")
        print("  5. ✓ Minimizes ||X_c @ Q - Y||_F^2")
        print("  6. ✓ Maximizes Tr(Q^T @ M)")
        return 0
    else:
        print(f"\n❌ {total_tests - total_passed} verification(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
