import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, csc_array, dia_array, linalg, coo_array, tril, triu, diags
from numba import njit
from scipy.sparse.linalg import spsolve_triangular


def Jacobi(A, b:np.ndarray, x0:np.ndarray, TOL, N):
    """
    Jacobi迭代法
    :param A:
    :param b:
    :param x0:
    :param TOL:
    :param N: 最大迭代次数
    :return:
    """
    x = x0.copy()
    n = A.shape[0]
    dig = A.diagonal().reshape(n, 1)
    for k in range(N):
        r = b - A.dot(x)
        delta_x = r/dig
        x += delta_x
        if np.linalg.norm(delta_x) < TOL:
            print(f"Jacobi iter num: {k}")
            return x

    print("Jacobi error")


def gauss_seidel(A, b, x0, max_iter, tol):
    n = len(b)
    x = x0.copy()
    A_csr = csr_array(A)  # 转换为CSR格式

    for iter_count in range(max_iter):
        x_prev = x.copy()
        for i in range(n):
            # 利用稀疏矩阵快速计算行内积
            sum1 = A_csr[i, :i].dot(x[:i])  # 新值部分
            sum2 = A_csr[i, i + 1:].dot(x_prev[i + 1:])  # 旧值部分
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        # 检查收敛
        if np.max(np.abs(x - x_prev)) < tol:
            return x
    return x  # 未收敛


def gauss_seidel1(A, b, x0, max_iter=1000, tol=1e-7):
    """
    使用Gauss-Seidel迭代法求解线性方程组Ax = b。

    参数:
        A (csc_matrix): 稀疏系数矩阵（CSC格式）。
        b (numpy.ndarray): 右侧向量。
        x0 (numpy.ndarray): 初始猜测解向量。
        max_iter (int): 最大迭代次数。
        tol (float): 收敛容差。

    返回:
        x (numpy.ndarray): 近似解向量。
    """
    # 分解A为下三角矩阵D-L和严格上三角矩阵U（取反）
    D_L = tril(A, format='csc')  # D - L为下三角矩阵（包括对角线）
    D_L.indices = D_L.indices.astype(np.int32)
    D_L.indptr = D_L.indptr.astype(np.int32)

    U = -triu(A, k=1, format='csc')  # 严格上三角部分取反

    x = x0.copy().astype(float)  # 确保使用浮点数
    b = b.astype(float)

    for it in range(max_iter):
        # 计算Ux^(k)
        Ux = U.dot(x)
        # 构造右侧向量: Ux^(k) + b
        rhs = Ux + b

        # 解下三角方程组 (D-L)x^(k+1) = rhs
        x_new = spsolve_triangular(D_L, rhs, lower=True)

        # 计算残差范数
        residual = np.linalg.norm(A.dot(x_new) - b)
        if residual < tol:
            x = x_new
            print(f"G-S iter num: {it}")
            return x
        x = x_new
    print("G-S not converged")
    return x

def SOR(A, b, x0, w,max_iter=1000, tol=1e-7):
    """
    SOR迭代法
    :param A: 稀疏系数矩阵（CSC格式）
    :param b: 右侧向量
    :param x0: 初始猜测解向量
    :param w: 松弛参数
    :param max_iter: 最大迭代次数
    :param tol: 收敛容差
    :return: 近似解向量
    """
    D = diags(A.diagonal(), format="csc")
    L = -tril(A, k=-1, format="csc")
    U = -triu(A, k=1, format="csc")

    M1 = D - w * L
    M2 = (1 - w) * D + w * U

    for it in range(max_iter):
        rhs = M2.dot(x0) + w*b

        x = spsolve_triangular(M1, rhs, lower=True)

        residual = np.linalg.norm(A.dot(x) - b)
        if residual < tol:
            print(f"SOR iter num: {it}")
            return x
        x0 = x.copy()
    print("SOR not converged")
    return x0