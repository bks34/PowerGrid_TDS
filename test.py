import numpy as np
from algorithm.other_solver import Jacobi, gauss_seidel, gauss_seidel1, SOR
import time

n = 1000
A = n*np.eye(n)+np.random.uniform(-1.0, 1.0, size=(n, n))
b = np.ones((n, 1))

t1 = time.time()
x_true = np.linalg.solve(A, b)
t2 = time.time()
x_Jacobi = Jacobi(A=A, b=b, x0=np.zeros((n, 1)), TOL=1e-7, N=10*n)
t3 = time.time()
# x_GS = gauss_seidel(A=A, b=b, x0=np.zeros((n, 1)), tol=1e-7, max_iter=10*n)
t4 = time.time()
x_GS1 = gauss_seidel1(A=A, b=b, x0=np.zeros((n, 1)), tol=1e-7, max_iter=10*n)
t5 = time.time()
x_SOR = SOR(A=A, b=b, x0=np.zeros((n, 1)), w=1.01,tol=1e-7, max_iter=10*n)
t6 = time.time()

print(f"normal used {t2-t1}s")
print(f"Jacobi used {t3-t2}s")
# print(f"G-S used {t4-t3}")
print(f"G-S1 used {t5 - t4}")
print(f"SOR used {t6 - t5}")
print(f"error: {np.linalg.norm(x_true-x_SOR)}")

