from numba import njit
import numpy as np
from scipy.sparse import csc_array

@njit
def incomplete_cholesky(A, alpha=0.0):
    """
    不完全 Cholesky 分解 (ICC) 实现（稠密矩阵版本）

    参数:
        A (np.ndarray): 对称正定矩阵 (n x n)
        alpha (float): MIC修正参数 (alpha >= 0)

    返回:
        L (np.ndarray): 下三角矩阵，满足 A ≈ L * L^T
    """
    n = A.shape[0]
    L = np.zeros_like(A)  # 初始化下三角矩阵

    for i in range(n):
        # 对角线元素
        L_ii = A[i, i]
        for k in range(i):
            L_ii -= L[i, k] ** 2

        # MIC修正项
        sum_sq = 0.0
        for k in range(i):
            sum_sq += L[i, k] ** 2
        L_ii += alpha * sum_sq

        if L_ii <= 0:
            raise ValueError("分解失败：矩阵非正定或需要更大修正参数 alpha")

        L[i, i] = np.sqrt(L_ii)

        # 计算第i行的非对角元素
        for j in range(i + 1, n):
            if A[j, i] == 0:  # 仅保留原始非零结构
                continue
            val = A[j, i]
            for k in range(i):
                val -= L[j, k] * L[i, k]
            L[j, i] = val / L[i, i]

    return L

def incomplete_cholesky_s(A:csc_array, drop_tol=1e-4):
    """
    实现 IC(0) 不完全 Cholesky 分解（对角线调整确保正定性）

    参数：
        A : scipy.sparse.csc_matrix
            对称正定稀疏矩阵（CSC格式）
        drop_tol : float
            小元素丢弃阈值（增强稀疏性）

    返回：
        L : scipy.sparse.csc_matrix
            下三角因子（可能包含调整后的对角线）
    """
    n = A.shape[0]
    L = A.copy()    # 复制A的下三角结构

    # 遍历每一列
    for j in range(n):
        # 处理对角线元素
        start = L.indptr[j]
        end = L.indptr[j + 1]
        diag_index = np.where(L.indices[start:end] == j)[0]
        if len(diag_index) == 0:
            raise ValueError("矩阵缺少对角线元素，无法进行Cholesky分解")
        diag_pos = start + diag_index[0]

        # 计算平方和修正项
        sum_sq = 0.0
        for i in range(start, diag_pos):
            row = L.indices[i]
            sum_sq += L.data[i] ** 2

        # 调整对角线避免非正定
        adjusted_diag = np.sqrt(np.abs(A[j, j] - sum_sq))
        L.data[diag_pos] = adjusted_diag

        # 处理非对角线元素
        for i in range(diag_pos + 1, end):
            row = L.indices[i]
            sum_ik = 0.0
            # 遍历L的行row和列j的共同非零元素
            row_start = L.indptr[row]
            row_end = L.indptr[row + 1]
            col_start = L.indptr[j]
            col_end = L.indptr[j + 1]

            k_row = L.indices[row_start:row_end]
            k_col = L.indices[col_start:col_end]

            # 共同列索引（k < j）
            common_ks = np.intersect1d(k_row, k_col, assume_unique=True)
            for k in common_ks:
                if k < j:
                    sum_ik += L[row, k] * L[j, k]

            # 计算L[i,j]
            val = (A[row, j] - sum_ik) / adjusted_diag
            # 应用丢弃策略
            if abs(val) < drop_tol:
                val = 0.0
            L.data[i] = val

        # 移除本列中被置零的元素
        mask = L.data[start:end] != 0.0
        new_indices = L.indices[start:end][mask]
        new_data = L.data[start:end][mask]
        L.indptr[j + 1] = L.indptr[j] + len(new_data)
        if j < n - 1:
            L.indices = np.concatenate((L.indices[:start], new_indices, L.indices[end:]))
            L.data = np.concatenate((L.data[:start], new_data, L.data[end:]))

    L.eliminate_zeros()
    return L