import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, csc_array, dia_array, linalg, coo_array, tril, triu, diags
from numba import njit
from scipy.sparse.linalg import spsolve_triangular


def Jacobi(A, b:np.ndarray, x0:np.ndarray, tol, max_iter):
    """
    Jacobi迭代法
    :param A: 稀疏系数矩阵（CSC格式）
    :param b: 右侧向量
    :param x0: 初始猜测解向量
    :param max_iter: 最大迭代次数
    :param tol: 收敛容差
    :return: 近似解向量
    """
    A = csr_array(A)
    n = A.shape[0]

    # 预提取稀疏矩阵结构
    indptr = A.indptr.astype(np.int32)
    indices = A.indices.astype(np.int32)
    data = A.data.astype(np.float64)

    diag = A.diagonal()

    # 初始化解向量
    x = x0.copy().astype(np.float64)

    for it in range(1, max_iter + 1):
        x_prev = x.copy()
        x = Jacobi_core(n=n, b=b.flatten(), x=x.flatten(), x_prev=x_prev.flatten(),
                        indptr=indptr, indices=indices, data=data, diag=diag)
        # 收敛判断
        residual = b - A.dot(x)
        norm_val = np.linalg.norm(residual)

        if norm_val < tol:
            print(f"Jacobi iter num: {it}")
            return x

    print("Jacobi not converged")
    return x

@njit
def Jacobi_core(n, b, x, x_prev, indptr, indices, data, diag):
    for i in range(n):
        # 获取当前行的非零元素
        start = indptr[i]
        end = indptr[i + 1]

        # 计算非对角线元素的加权和
        sum_val = 0.0
        for j in range(start, end):
            col = indices[j]
            if col == i:  # 排除对角线元素
                continue
            else:
                sum_val += data[j] * x_prev[col]
        # 更新当前分量
        x[i] = (b[i] - sum_val) / diag[i]

    return x.reshape(n, 1)


def Jacobi1(A, b:np.ndarray, x0:np.ndarray, tol, max_iter):
    """
    Jacobi迭代法(向量格式)
    :param A: 稀疏系数矩阵（CSC格式）
    :param b: 右侧向量
    :param x0: 初始猜测解向量
    :param max_iter: 最大迭代次数
    :param tol: 收敛容差
    :return: 近似解向量
    """
    x = x0.copy()
    n = A.shape[0]
    dig = A.diagonal().reshape(n, 1)
    for k in range(1, max_iter + 1):
        r = b - A.dot(x)
        delta_x = r/dig
        x += delta_x
        if np.linalg.norm(delta_x) < tol:
            print(f"Jacobi iter num: {k}")
            return x

    print("Jacobi error")


def gauss_seidel(A, b, x0=None, max_iter=1000, tol=1e-5):
    """
    Gauss-seidel迭代
    :param A: 稀疏系数矩阵（CSC格式）
    :param b: 右侧向量
    :param x0: 初始猜测解向量
    :param max_iter: 最大迭代次数
    :param tol: 收敛容差
    :return: 近似解向量
    """
    # 转换为CSR格式以支持高效的行访问
    A = csr_array(A)
    n = A.shape[0]

    # 预提取稀疏矩阵结构
    indptr = A.indptr.astype(np.int32)
    indices = A.indices.astype(np.int32)
    data = A.data.astype(np.float64)

    diag = A.diagonal()

    # 初始化解向量
    x = x0.copy().astype(np.float64)

    # 主迭代循环
    for it in range(1, max_iter+1):
        x_prev = x.copy()

        # 使用numba优化迭代
        x = gauss_seidel_core(n=n, b=b.flatten(), x=x.flatten(), x_prev=x_prev.flatten(),
                              indptr=indptr, indices=indices, data=data, diag=diag)
        # 收敛判断
        residual = b - A.dot(x)
        norm_val = np.linalg.norm(residual)

        if norm_val < tol:
            print(f"G-S iter num: {it}")
            return x

    print("G-S not converged")
    return x

@njit
def gauss_seidel_core(n, b, x, x_prev, indptr, indices, data, diag):
    """
    使用numba优化了循环的G-S核心部分
    """
    for i in range(n):
        # 获取当前行的非零元素
        start = indptr[i]
        end = indptr[i + 1]

        # 计算非对角线元素的加权和
        sum_val = 0.0
        for j in range(start, end):
            col = indices[j]
            if col == i:  # 排除对角线元素
                continue
            elif col < i:
                sum_val += data[j] * x[col]
            else:
                sum_val += data[j] * x_prev[col]

        # 更新当前分量
        x[i] = (b[i] - sum_val) / diag[i]
    return x.reshape(n, 1)



def gauss_seidel1(A, b, x0, max_iter=1000, tol=1e-7):
    """
    Gauss-seidel迭代法(向量格式)
    :param A: 稀疏系数矩阵（CSC格式）
    :param b: 右侧向量
    :param x0: 初始猜测解向量
    :param max_iter: 最大迭代次数
    :param tol: 收敛容差
    :return: 近似解向量
    """
    # 分解A为下三角矩阵D-L和严格上三角矩阵U（取反）
    D_L = tril(A, format='csc')  # D - L为下三角矩阵（包括对角线）
    D_L.indices = D_L.indices.astype(np.int32)
    D_L.indptr = D_L.indptr.astype(np.int32)

    U = -triu(A, k=1, format='csc')  # 严格上三角部分取反

    x = x0.copy().astype(float)  # 确保使用浮点数
    b = b.astype(float)

    for it in range(1, max_iter+1):
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


def SOR(A, b, x0, w, max_iter=1000, tol=1e-7):
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
    # 转换为CSR格式以支持高效的行访问
    A = csr_array(A)
    n = A.shape[0]

    # 预提取稀疏矩阵结构
    indptr = A.indptr.astype(np.int32)
    indices = A.indices.astype(np.int32)
    data = A.data.astype(np.float64)

    diag = A.diagonal()

    # 初始化解向量
    x = x0.copy().astype(np.float64)

    # 主迭代循环
    for it in range(1, max_iter + 1):
        x_prev = x.copy()

        # 使用numba优化迭代
        x = SOR_core(n=n, b=b.flatten(), x=x.flatten(), x_prev=x_prev.flatten(),
                              indptr=indptr, indices=indices, data=data, diag=diag, w=w)
        # 收敛判断
        residual = b - A.dot(x)
        norm_val = np.linalg.norm(residual)

        if norm_val < tol:
            print(f"SOR iter num: {it}")
            return x

    print("SOR not converged")
    return x

@njit
def SOR_core(n, b, x, x_prev, indptr, indices, data, diag, w):
    """
    使用numba优化了循环的SOR核心部分
    """
    for i in range(n):
        # 获取当前行的非零元素
        start = indptr[i]
        end = indptr[i + 1]

        # 计算非对角线元素的加权和
        sum_val = 0.0
        for j in range(start, end):
            col = indices[j]
            if col == i:  # 排除对角线元素
                continue
            elif col < i:
                sum_val += data[j] * x[col]
            else:
                sum_val += data[j] * x_prev[col]

        # 更新当前分量
        x[i] = (1 - w) * x_prev[i] + w * (b[i] - sum_val) / diag[i]
    return x.reshape(n, 1)


def SOR1(A, b, x0, w, max_iter=1000, tol=1e-7):
    """
    SOR迭代法(向量格式)
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

    for it in range(1, max_iter+1):
        rhs = M2.dot(x0) + w*b

        x = spsolve_triangular(M1, rhs, lower=True)

        residual = np.linalg.norm(A.dot(x) - b)
        if residual < tol:
            print(f"SOR iter num: {it}")
            return x
        x0 = x.copy()
    print("SOR not converged")
    return x0