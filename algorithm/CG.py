"""
共轭梯度法与预处理共轭梯度法
稀疏矩阵版本
"""
import time

from numba import njit
import numpy as np
from numpy.linalg import eig
from scipy.linalg import solve_triangular
from scipy.sparse import csr_array, csc_array, dia_array
from scipy.sparse.linalg import spilu

def is_positive_define(matrix):
    matrix = matrix.toarray()
    e = eig(matrix)[0]
    return np.all(e > 0)

def cg(A:csc_array, b:np.ndarray, x:np.ndarray, TOL:float):
    """
    共轭梯度法（稀疏矩阵版本）
    :param A: 系数矩阵
    :param b: 常数项
    :param x: 初值
    :param TOL: 误差
    :return: 近似解
    """
    n = A.shape[0]
    IterMax = 1000*n
    r = b-A.dot(x)
    beta = np.linalg.norm(r)
    pho_prev = 0
    p = r
    for m in range(1, IterMax+1):
        pho = r.T@r
        if m>1:
            mu = pho/pho_prev
            p = r + mu*p
        q = A.dot(p)
        e = pho/(p.T@q)
        x = x + e*p
        r = r - e*q
        relres = np.linalg.norm(r)/beta
        if relres < TOL:
            print(f"iter num:{m}")
            return x
        pho_prev =pho
    print("cg error!")

def pcg(A:csc_array, b: np.ndarray, x: np.ndarray, M, TOL: float):
    """
    预处理CG算法（稀疏矩阵版本）
    :param A: 系数矩阵
    :param b: 常数项
    :param x: 初值
    :param M: 预处理子的逆
    :param TOL: 精度要求
    :return: 近似解
    """
    n = A.shape[0]
    IterMax = 1000*n
    r = b - A.dot(x)
    beta = np.linalg.norm(r)
    z = M @ r
    p = z
    pho = r.T@z
    for k in range(1, IterMax+1):
        q = A.dot(p)
        xi = pho/(p.T@q)
        x = x + xi*p
        r = r - xi*q
        relres = np.linalg.norm(r)/beta
        if relres < TOL:
            print(f"iter num:{k}")
            return x, k
        pho0 = pho
        z = M @ r
        pho = r.T@z
        mu = pho/pho0
        p = z + mu*p
    print("pcg solve error!")
