import numpy as np
from scipy.linalg import svd

# 构造矩阵
A = np.array([[1, 2],
              [3, 4],
              [1, 2],
              [1, 2]])

Au = np.array([[1, 2],
               [3, 4]])

# 构造选择矩阵S
S = np.array([[1, 0],
              [0, 1],
              [1, 0],
              [1, 0]])

# 验证 A = S @ Au
print("验证 A = S @ Au:")
print("A - S @ Au =", np.max(np.abs(A - S @ Au)))

# 1. 对A直接进行SVD
U, Sigma, Vt = svd(A, full_matrices=False)
print("\nA的SVD:")
print("Sigma:", Sigma)
print("Vt:\n", Vt)

# 2. 对Au进行SVD
U_u, Sigma_u, Vt_u = svd(Au, full_matrices=False)
print("\nAu的SVD:")
print("Sigma_u:", Sigma_u)
print("Vt_u:\n", Vt_u)

# 3. 计算权重矩阵D
D = np.diag([3, 1])  # 重复次数: [3, 1]
print("\n权重矩阵D:\n", D)

# 4. 计算加权矩阵 A_w = D^{1/2} @ Au
D_sqrt = np.sqrt(D)  # D^{1/2}
A_w = D_sqrt @ Au
print("\n加权矩阵 A_w = D^{1/2} @ Au:")
print(A_w)

# 5. 对A_w进行SVD
U_w, Sigma_w, Vt_w = svd(A_w, full_matrices=False)
print("\nA_w的SVD:")
print("Sigma_w:", Sigma_w)
print("Vt_w:\n", Vt_w)

# 6. 从A_w的SVD推导A的SVD
# A = S @ Au = S @ D^{-1/2} @ A_w
# A = S @ D^{-1/2} @ U_w @ diag(Sigma_w) @ Vt_w

# 计算 U_from_w = S @ D^{-1/2} @ U_w
D_inv_sqrt = np.linalg.inv(D_sqrt)
U_from_w = S @ D_inv_sqrt @ U_w

print("\n从A_w的SVD推导的A的SVD:")
print("Sigma (从A_w):", Sigma_w)
print("Vt (从A_w):\n", Vt_w)
print("U (计算得到):\n", U_from_w)

# 7. 验证重构
A_recon_from_w = U_from_w @ np.diag(Sigma_w) @ Vt_w
print("\n用A_w的SVD重构A:")
print("重构A:\n", A_recon_from_w)
print("重构误差:", np.max(np.abs(A - A_recon_from_w)))

# 8. 比较奇异值和右奇异向量
print("\n比较奇异值:")
print("A的Sigma:", Sigma)
print("A_w的Sigma_w:", Sigma_w)
print("差异:", np.max(np.abs(Sigma - Sigma_w)))

print("\n比较Vt（调整符号）:")
# 调整符号以匹配
sign = np.sign(Vt[0, :] * Vt_w[0, :])
Vt_w_adjusted = Vt_w * sign
print("Vt:\n", Vt)
print("调整后的Vt_w:\n", Vt_w_adjusted)
print("差异:", np.max(np.abs(Vt - Vt_w_adjusted)))

print("\n比较U（调整符号和缩放）:")
# 调整U_from_w的符号以匹配U
sign_u = np.sign(U[0, :] * U_from_w[0, :])
U_from_w_adjusted = U_from_w * sign_u
print("U:\n", U)
print("调整后的U_from_w:\n", U_from_w_adjusted)
print("差异:", np.max(np.abs(U - U_from_w_adjusted)))

# 9. 验证特征值关系
print("\n" + "="*50)
print("特征值关系验证:")

# A^T A 的特征值
ATA = A.T @ A
eigvals_A = np.linalg.eigvals(ATA)
eigvals_A_sorted = np.sort(eigvals_A)[::-1]
print("A^T A 的特征值:", eigvals_A_sorted)

# Au^T D Au 的特征值
AuT_D_Au = Au.T @ D @ Au
eigvals_AuT_D_Au = np.linalg.eigvals(AuT_D_Au)
eigvals_AuT_D_Au_sorted = np.sort(eigvals_AuT_D_Au)[::-1]
print("Au^T D Au 的特征值:", eigvals_AuT_D_Au_sorted)

# Au^T Au 的特征值
AuT_Au = Au.T @ Au
eigvals_AuT_Au = np.linalg.eigvals(AuT_Au)
eigvals_AuT_Au_sorted = np.sort(eigvals_AuT_Au)[::-1]
print("Au^T Au 的特征值:", eigvals_AuT_Au_sorted)

# 特征值的比值
print("\n特征值比值:")
print("A^T A / (Au^T D Au):", eigvals_A_sorted / eigvals_AuT_D_Au_sorted)
print("A^T A / (Au^T Au):", eigvals_A_sorted / eigvals_AuT_Au_sorted)