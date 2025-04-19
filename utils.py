import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, csc_array, dia_array, linalg, coo_array


def calculate_G2(B:csc_array, theta_t_1:np.ndarray,
                 w_t_1:np.ndarray, w:np.ndarray, E_t_1:np.ndarray,
                 dt, D:np.ndarray, M:np.ndarray, P:np.ndarray):
    """
    计算G2
    :param B:
    :param theta_t_1: t+1时刻的theta
    :param w_t_1:
    :param w:
    :param E_t_1:
    :param dt:
    :param D:
    :param M:
    :param P:
    :return: G2
    """
    B_coo = B.tocoo()
    rows = B_coo.row
    cols = B_coo.col
    data = B_coo.data

    # 计算所有非零元素的正弦项
    sin_vals = np.sin(theta_t_1.flatten()[rows] - theta_t_1.flatten()[cols])
    weighted_sin = data * sin_vals

    # 计算E_i*E_j
    product_E = E_t_1.flatten()[rows]*E_t_1.flatten()[cols]
    tmp = product_E * weighted_sin

    # 求和
    n = B.shape[0]
    tmp_sum = np.zeros(n)
    np.add.at(tmp_sum, rows, tmp)
    tmp_sum = tmp_sum.reshape(n,1)

    # G2
    G2 = dt/M*tmp_sum + (1+dt*D/M)*w_t_1 - dt*P/M - w
    return G2


def calculate_G3(B:csc_array, theta_t_1:np.ndarray,
                 E_t_1:np.ndarray, E:np.ndarray,
                 dt, X:np.ndarray, T:np.ndarray, Ef:np.ndarray):
    """
    计算G3
    :param B:
    :param theta_t_1:
    :param E_t_1:
    :param E:
    :param dt:
    :param X:
    :param T:
    :param Ef:
    :return:
    """
    B_coo = B.tocoo()
    rows = B_coo.row
    cols = B_coo.col
    data = B_coo.data

    # 计算所有非零元素的余弦项
    cos_vals = np.cos(theta_t_1.flatten()[rows] - theta_t_1.flatten()[cols])
    weighted_cos = data * cos_vals

    tmp = E_t_1.flatten()[cols] * weighted_cos

    # 求和
    n = B.shape[0]
    tmp_sum = np.zeros(n)
    np.add.at(tmp_sum, rows, tmp)
    tmp_sum = tmp_sum.reshape(n, 1)

    # G3
    G3 = (1+dt/T)*E_t_1 - dt*X/T*tmp_sum - dt*Ef/T - E

    return G3


def calculate_A(B:csc_array, theta_t_1:np.ndarray, E_t_1:np.ndarray):
    """
    计算A
    :param B:
    :param theta_t_1:
    :param E_t_1:
    :return: A
    """
    # 提取COO格式的非零元素信息
    B_coo = B.tocoo()
    rows = B_coo.row
    cols = B_coo.col
    data = B_coo.data

    # 计算所有非零元素的余弦项
    cos_vals = np.cos(theta_t_1.flatten()[rows] - theta_t_1.flatten()[cols])
    weighted_cos = data * cos_vals


    # E_i*E_j
    product_E = E_t_1.flatten()[rows] * E_t_1.flatten()[cols]
    values = product_E*weighted_cos


    # 过滤非对角元素并累加对角线和
    mask = (rows != cols)
    rows_nondiag = rows[mask]
    cols_nondiag = cols[mask]
    nondiag_values = values[mask]

    n = B.shape[0]
    diag_sum = np.zeros(n)
    np.add.at(diag_sum, rows_nondiag, nondiag_values)

    # 构造非对角元素数据
    nondiag_data = -nondiag_values

    # 合并所有元素数据
    diag_rows = np.arange(n)
    all_rows = np.concatenate([rows_nondiag, diag_rows])
    all_cols = np.concatenate([cols_nondiag, diag_rows])
    all_data = np.concatenate([nondiag_data, diag_sum])

    # 生成CSC矩阵
    A = coo_array((all_data, (all_rows, all_cols)), shape=B.shape).tocsc()
    return A

def calculate_C(B:csc_array, theta_t_1:np.ndarray, E_t_1:np.ndarray):
    """
    计算矩阵C
    :param B:
    :param theta_t_1:
    :param E_t_1:
    :return:
    """
    # 提取COO格式的非零元素信息
    B_coo = B.tocoo()
    rows = B_coo.row
    cols = B_coo.col
    data = B_coo.data

    # 计算所有非零元素的正弦项
    sin_vals = np.sin(theta_t_1.flatten()[rows] - theta_t_1.flatten()[cols])
    weighted_sin = data * sin_vals

    values = E_t_1.flatten()[rows] * weighted_sin

    # 过滤非对角元素并累加对角线和
    mask = (rows != cols)
    rows_nondiag = rows[mask]
    cols_nondiag = cols[mask]
    nondiag_values = values[mask]

    n = B.shape[0]
    diag_sum = np.zeros(n)
    np.add.at(diag_sum, cols_nondiag, nondiag_values)
    diag_sum = -diag_sum

    # 合并所有元素数据
    diag_rows = np.arange(n)
    all_rows = np.concatenate([rows_nondiag, diag_rows])
    all_cols = np.concatenate([cols_nondiag, diag_rows])
    all_data = np.concatenate([nondiag_values, diag_sum])

    # 生成CSC矩阵
    C = coo_array((all_data, (all_rows, all_cols)), shape=B.shape).tocsc()
    return C

def calculate_H(B:csc_array, theta_t_1:np.ndarray):
    """
    计算矩阵H
    :param B:
    :param theta_t_1:
    :return:
    """
    # 提取COO格式的非零元素信息
    B_coo = B.tocoo()
    rows = B_coo.row
    cols = B_coo.col
    data = B_coo.data

    # 计算所有非零元素的余弦项
    cos_vals = np.cos(theta_t_1.flatten()[rows] - theta_t_1.flatten()[cols])
    weighted_cos = data * cos_vals

    # 生成CSC矩阵
    H = coo_array((weighted_cos, (rows, cols)), shape=B.shape).tocsc()
    return H


def plot_matrix_spectrum(matrix, title="Matrix Spectrum"):
    """
    绘制矩阵的特征值在复平面上的分布（谱）。

    参数:
        matrix (np.ndarray): 输入的方阵
        title (str): 图表标题
    """
    # 检查输入是否为方阵
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("输入必须是一个方阵")

    # 计算特征值
    eigenvalues = np.linalg.eigvals(matrix)

    print(np.max(np.absolute(eigenvalues)))

    # 分离实部和虚部
    real_part = eigenvalues.real
    imag_part = eigenvalues.imag

    # 绘制复平面上的特征值
    plt.figure(figsize=(8, 6))
    plt.scatter(real_part, imag_part, color='blue', alpha=0.6, label='Eigenvalues')

    # 添加坐标轴和单位圆
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # 单位圆（可选）
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=1, label='Unit Circle')

    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title(title)
    plt.axis('equal')  # 保证坐标轴比例一致
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

