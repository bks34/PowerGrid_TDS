"""
TDS(时域分析)
稀疏矩阵版本
"""
import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os

from algorithm.CG import cg, pcg, is_positive_define
from algorithm.other_solver import Jacobi, gauss_seidel1, SOR
from power_system import PowerSystem
from utils import calculate_G2, calculate_G3, calculate_A, calculate_C, calculate_H, plot_matrix_spectrum

from scipy.sparse import csr_array, csc_array, dia_array, linalg


class Settings:
    """
    储存一些要用的参数
    """
    def __init__(self, dt, total_time, TOL_Newton, TOL_CG):
        self.dt = dt  # 时间步进长度
        self.total_time = total_time  # 模拟的最长时间
        self.TOL_Newton = TOL_Newton  # 牛顿迭代的误差
        self.TOL_CG = TOL_CG  # 子空间迭代的误差


class PreConditioner2Controller:
    """
    储存预处理子的逆，控制何时进行不完全cholesky分解的参数
    """
    def __init__(self, size):
        """

        :param size: 预处理子的规模，即系数矩阵的大小
        """
        self.inv_PreConditioner2 = csc_array((size, size))
        self.calculated = False
        # 上次迭代的次数
        self.iter_num_pre = 0
        # 上次迭代时t的值
        self.t_pre = 0
        # 一次pcg的时间
        self.time_pcg = 0
        # 一次不完全分解的时间
        self.time_decompose = 0
        # t迭代一次，牛顿迭代大概的次数
        self.iter_num_newton = 4

def show_result(t, theta:np.ndarray, w:np.ndarray, E:np.ndarray, i:int):
    os.system("clear")
    print("Result in t:", t)
    print(f"theta{i+1}:", theta[i])
    print(f"w{i+1}:", w[i])
    print(f"E{i+1}:", E[i])


def solve(size, theta:np.ndarray, w:np.ndarray,
          E:np.ndarray, M:dia_array, D:dia_array, T:dia_array, X:dia_array,
          P:np.ndarray, Ef:np.ndarray, B:csc_array,
          dt, total_time, TOL_Newton, TOL_CG, method:str = 'cg', error_ctrl:bool = False):
    """
    进行时域分析
    :param size: 网格规模
    :param theta: 各节点初始角度
    :param w: 各节点初始角速度
    :param E: 各节点初始电压
    :param M: 惯性
    :param D: 阻尼
    :param T: 瞬态电压动态的弛豫时间
    :param X: d轴上静态电抗X与瞬态电抗X‘的差值
    :param P: 功率
    :param Ef:
    :param B: 耦合强度
    :param dt: 时间步进长度
    :param total_time: 模拟的最长时间
    :param TOL_Newton: 牛顿迭代的误差
    :param TOL_CG: 共轭梯度法的误差控制
    :param method: 使用哪种求解方法,“normal”,"Jacobi","G-S","SOR","cg","pcg1"或"pcg2"
    :param error_ctrl: 是否进行内外误差控制
    :return:
    """
    t = 0.0  # 当前时间

    # 用于画图
    plot_t = []
    plot_theta1 = []
    plot_w1 = []
    plot_E1 = []
    plot_value_name = ["theta1", "w1", "E1"]

    y23 = np.zeros((2*size, 1))

    # 用来观察出现故障后的变化
    changed = False

    # 用于记录上一时间y的数量级
    error_newton_pre = 1
    num_of_newton_pre = 5

    num_of_newton_total = 0

    # pcg2(待修改)
    pre2_ctrl = PreConditioner2Controller(size)

    # pcg1(待修改)
    inv_X = X.copy()
    inv_X.data = 1 / inv_X.data
    inv_PreConditioner1_data1 = (M + dt*D).data.flatten()
    inv_PreConditioner1_data2 = (1/dt*inv_X*T + inv_X).data.flatten()
    inv_PreConditioner1 = dia_array((1/np.concatenate((inv_PreConditioner1_data1, inv_PreConditioner1_data2), axis=0), 0),
                                    shape=(2*size, 2*size))

    # 对时间t的主循环
    while t < total_time:
        # # 用于模拟发生故障
        # if (not changed) and t > total_time/2:
        #     P[0] -= 0.5
        #     P[1] += 0.5
        #     changed = True

        # 设置这一时刻t的初始TOL_CG
        if error_ctrl:
            TOL_CG = error_newton_pre /100
            if TOL_CG <= TOL_Newton / 100:
                TOL_CG = TOL_Newton / 100
            if num_of_newton_pre <= 2:
                TOL_CG = TOL_Newton / 100

        # t+1时刻的初值取t时刻的值
        theta_t_1 = theta.copy()
        w_t_1 = w.copy()
        E_t_1 = E.copy()

        # 记录所用Newton迭代次数
        num_of_newton = 0
        # Newton迭代循环
        while True:
            num_of_newton += 1

            # 计算需要用到的各个矩阵
            # 计算G1,G2,G3
            t_1 = time.time()
            G1 = theta_t_1 - dt * w_t_1 - theta
            t_2 = time.time()
            G2 = calculate_G2(B=B, theta_t_1=theta_t_1, w_t_1=w_t_1, w=w,
                              E_t_1=E_t_1, dt=dt, D=D.data.reshape(size, 1),
                              M=M.data.reshape(size, 1), P=P)
            t_3 = time.time()
            G3 = calculate_G3(B=B, theta_t_1=theta_t_1, E_t_1=E_t_1, E=E,
                              dt=dt, X=X.data.reshape(size, 1), T=T.data.reshape(size, 1),
                              Ef=Ef)
            t_4 = time.time()
            # print(f"Calculate G1 used:{t_2-t_1}s")
            # print(f"Calculate G2 used:{t_3-t_2}s")
            # print(f"Calculate G3 used:{t_4-t_3}s")


            # 计算A,C,H
            t_1 = time.time()
            A = calculate_A(B=B, theta_t_1=theta_t_1, E_t_1=E_t_1)
            t_2 = time.time()
            C = calculate_C(B=B, theta_t_1=theta_t_1, E_t_1=E_t_1)
            t_3 = time.time()
            H = calculate_H(B=B, theta_t_1=theta_t_1)
            t_4 = time.time()
            # print(f"Calculate A used:{t_2 - t_1}s")
            # print(f"Calculate C used:{t_3 - t_2}s")
            # print(f"Calculate H used:{t_4 - t_3}s")

            if method == "normal":
                # 构建要求解的线性方程
                inv_M = M.copy()
                inv_M.data = 1/inv_M.data
                inv_T = T.copy()
                inv_T.data = 1/inv_T.data
                system = sp.vstack([
                    sp.hstack([sp.eye_array(size), -dt*sp.eye_array(size), csc_array(np.zeros((size, size)))]),
                    sp.hstack([dt*inv_M@A, sp.eye_array(size)+dt*inv_M*D, dt*inv_M@C]),
                    sp.hstack([dt*inv_T*X@C.T, csc_array(np.zeros((size,size))), sp.eye_array(size)+dt*inv_T*(sp.eye_array(size)-X@H)])
                ])
                system = csc_array(system)
                G = -np.vstack([G1, G2, G3])
                t_1 = time.time()
                y = sp.linalg.spsolve(system, G)
                t_2 = time.time()
                print(f"Calculate y used:{t_2 - t_1}s")
                y = y.reshape(3*size, 1)

            else:
                # 构建要求解的线性方程
                t_1 = time.time()
                system = sp.vstack([
                    sp.hstack([M + dt * D + dt * dt * A, dt * C]),
                    sp.hstack([dt * C.T, 1 / dt * inv_X * T + (inv_X - H)])
                ])
                system = csc_array(system)

                t_2 = time.time()
                b1 = dt * A @ G1 - M @ G2
                b2 = C.T @ G1 - 1 / dt * inv_X * T @ G3
                b = np.vstack((b1, b2))
                t_3 = time.time()
                # print(f"Calculate system used:{t_2 - t_1}s")
                # print(f"Calculate b used:{t_3 - t_2}s")

                if method == 'Jacobi':
                    print("\nJacobi")
                    t_1 = time.perf_counter()
                    y23 = Jacobi(A=system, b=b, x0=y23, TOL=TOL_CG, N=10*size)
                    t_2 = time.perf_counter()
                    print(f"Jacobi use {t_2 - t_1}s")

                elif method == 'G-S':
                    print("\nG-S")
                    t_1 = time.perf_counter()
                    y23 = gauss_seidel1(A=system, b=b, x0=y23, tol=TOL_CG, max_iter=10*size)
                    t_2 = time.perf_counter()
                    print(f"G-S use {t_2 - t_1}s")

                elif method == "SOR":
                    print("\nSOR")
                    t_1 = time.perf_counter()
                    y23 = SOR(A=system, b=b, x0=y23, w=1.0000093183721113, tol=TOL_CG, max_iter=10*size)
                    t_2 = time.perf_counter()
                    print(f"SOR use {t_2 - t_1}s")

                elif method == 'cg':
                    print("\ncg:")
                    t_cg_1 = time.perf_counter()
                    y23 = cg(A=system, b=b, x=y23, TOL=TOL_CG)
                    t_cg_2 = time.perf_counter()
                    print(f"cg use {t_cg_2 - t_cg_1}s")

                elif method == 'pcg1':
                    print("\npcg with preconditioner Jacobian:")
                    t_pcg1_1 = time.perf_counter()
                    y23, _ = pcg(A=system, b=b, x=y23, M=inv_PreConditioner1, TOL=TOL_CG)
                    t_pcg1_2 = time.perf_counter()
                    print(f"pcg with preconditioner Jacobian use {t_pcg1_2 - t_pcg1_1}s")

                elif method == 'pcg2':
                    print("\npcg with preconditioner L*L.T:")
                    t_cal_inv_Pre1 = time.perf_counter()
                    if not pre2_ctrl.calculated:
                        pre2_ctrl.inv_PreConditioner2 = csc_array(linalg.spilu(system).solve(np.eye(2 * size)))
                    t_cal_inv_Pre2 = time.perf_counter()
                    t_pcg2_1 = time.perf_counter()
                    y23, iter_num = pcg(A=system, b=b, x=y23, M=pre2_ctrl.inv_PreConditioner2, TOL=TOL_CG)
                    t_pcg2_2 = time.perf_counter()
                    if not pre2_ctrl.calculated:
                        pre2_ctrl.iter_num_pre = iter_num
                        pre2_ctrl.t_pre = t
                        pre2_ctrl.time_decompose = t_cal_inv_Pre2 - t_cal_inv_Pre1
                        pre2_ctrl.time_pcg = (t_pcg2_2 - t_pcg2_1) / pre2_ctrl.iter_num_pre
                        pre2_ctrl.calculated = True

                    if num_of_newton == 1 and (iter_num - pre2_ctrl.iter_num_pre) * pre2_ctrl.time_pcg * (
                            t - pre2_ctrl.t_pre) / dt * pre2_ctrl.iter_num_newton / 2 > pre2_ctrl.time_decompose:
                        pre2_ctrl.calculated = False
                    # if num_of_newton == 1 and iter_num > 2*pre2_ctrl.iter_num_pre:
                    #     pre2_ctrl.calculated = False

                    print(f"pcg with preconditioner L*L.T use {t_pcg2_2 - t_pcg2_1}s")
                    print(f"calculate L use {t_cal_inv_Pre2 - t_cal_inv_Pre1}")

                t_1 = time.time()
                y = np.append(dt * y23[0:size] - G1,
                              y23,
                              axis=0)
                t_2 = time.time()
                print(f"Calculate y used:{t_2 - t_1}s")

            theta_t_1 += y[0:size]
            w_t_1 += y[size:2*size]
            E_t_1 += y[2*size:3*size]
            # 如果y的模足够小

            error_newton = np.linalg.norm(y)
            print(str(num_of_newton) + ": " + str(error_newton) + " " + str(TOL_CG))
            print("theta:"+str(theta_t_1[0]), "w:"+str(w_t_1[0]),"E:"+str(E_t_1[0]))
            if error_ctrl:
                if num_of_newton == 1:
                    error_newton_pre = error_newton
                if num_of_newton == 2:
                    TOL_CG = TOL_Newton / 100
            if error_ctrl and error_newton < 10*TOL_CG:
                TOL_CG /= 1000
                if TOL_CG <= TOL_Newton/100:
                    TOL_CG = TOL_Newton/100
            if error_newton < TOL_Newton:
                pre2_ctrl.iter_num_newton = num_of_newton
                num_of_newton_pre = num_of_newton
                num_of_newton_total += num_of_newton
                print(num_of_newton)
                break
        t += dt
        # 更新theta与w
        theta = theta_t_1
        w = w_t_1
        E = E_t_1
        # 将该时刻的结果打印出来
        show_result(t=t, theta=theta, w=w, E=E, i=0)


        plot_t.append(t)
        plot_theta1.append(theta[0])
        plot_w1.append(w[0])
        plot_E1.append(E[0])

    # # 还原P,避免影响下一次试验
    # P[0] += 0.5
    # P[1] -= 0.5

    print(f"num of newton total:{num_of_newton_total}")

    # 画图，存储图像
    plot_values = [plot_theta1, plot_w1, plot_E1]
    for i in range(len(plot_values)):
        plt.plot(plot_t, plot_values[i])
        plt.title(method)
        plt.xlabel("t/s")
        plt.ylabel(plot_value_name[i])
        file_path = os.path.join("results/", method+"_" + plot_value_name[i] + ".png")
        plt.savefig(file_path)
        plt.close()

def main():
    """
    程序主入口
    :return:
    """
    setting = Settings(dt=0.01, total_time=10,
                       TOL_Newton=1e-5,
                       TOL_CG=1e-7)
    power_system = PowerSystem().read_from_matpower_file("cases/matpower/case3120sp.m")

    # t_normal_1 = time.perf_counter()
    # solve(size=power_system.node_size, theta=power_system.theta, w=power_system.w, E=power_system.E,
    #       M=power_system.M, D=power_system.D, T=power_system.T, X=power_system.X,
    #       P=power_system.P, Ef=power_system.Ef, B=power_system.B,
    #       dt=setting.dt, total_time=setting.total_time, TOL_Newton=setting.TOL_Newton, TOL_CG=setting.TOL_CG,
    #       method="normal", error_ctrl=False)
    # t_normal_2 = time.perf_counter()

    t_Jacobi_1 = time.perf_counter()
    solve(size=power_system.node_size, theta=power_system.theta, w=power_system.w, E=power_system.E,
          M=power_system.M, D=power_system.D, T=power_system.T, X=power_system.X,
          P=power_system.P, Ef=power_system.Ef, B=power_system.B,
          dt=setting.dt, total_time=setting.total_time, TOL_Newton=setting.TOL_Newton, TOL_CG=setting.TOL_CG,
          method="Jacobi", error_ctrl=True)
    t_Jacobi_2 = time.perf_counter()

    t_GS_1 = time.perf_counter()
    solve(size=power_system.node_size, theta=power_system.theta, w=power_system.w, E=power_system.E,
          M=power_system.M, D=power_system.D, T=power_system.T, X=power_system.X,
          P=power_system.P, Ef=power_system.Ef, B=power_system.B,
          dt=setting.dt, total_time=setting.total_time, TOL_Newton=setting.TOL_Newton, TOL_CG=setting.TOL_CG,
          method="G-S", error_ctrl=True)
    t_GS_2 = time.perf_counter()

    t_SOR_1 = time.perf_counter()
    solve(size=power_system.node_size, theta=power_system.theta, w=power_system.w, E=power_system.E,
          M=power_system.M, D=power_system.D, T=power_system.T, X=power_system.X,
          P=power_system.P, Ef=power_system.Ef, B=power_system.B,
          dt=setting.dt, total_time=setting.total_time, TOL_Newton=setting.TOL_Newton, TOL_CG=setting.TOL_CG,
          method="SOR", error_ctrl=True)
    t_SOR_2 = time.perf_counter()

    # t_cg_1 = time.perf_counter()
    # solve(size=power_system.node_size, theta=power_system.theta, w=power_system.w, E=power_system.E,
    #       M=power_system.M, D=power_system.D, T=power_system.T, X=power_system.X,
    #       P=power_system.P, Ef=power_system.Ef, B=power_system.B,
    #       dt=setting.dt, total_time=setting.total_time, TOL_Newton=setting.TOL_Newton, TOL_CG=setting.TOL_CG,
    #       method="cg", error_ctrl=True)
    # t_cg_2 = time.perf_counter()
    #
    # t_pcg1_1 = time.perf_counter()
    # solve(size=power_system.node_size, theta=power_system.theta, w=power_system.w, E=power_system.E,
    #       M=power_system.M, D=power_system.D, T=power_system.T, X=power_system.X,
    #       P=power_system.P, Ef=power_system.Ef, B=power_system.B,
    #       dt=setting.dt, total_time=setting.total_time, TOL_Newton=setting.TOL_Newton, TOL_CG=setting.TOL_CG,
    #       method="pcg1", error_ctrl=True)
    # t_pcg1_2 = time.perf_counter()

    # t_pcg2_1 = time.perf_counter()
    # solve(size=power_system.node_size, theta=power_system.theta, w=power_system.w, E=power_system.E,
    #       M=power_system.M, D=power_system.D, T=power_system.T, X=power_system.X,
    #       P=power_system.P, Ef=power_system.Ef, B=power_system.B,
    #       dt=setting.dt, total_time=setting.total_time, TOL_Newton=setting.TOL_Newton, TOL_CG=setting.TOL_CG,
    #       method="pcg2", error_ctrl=True)
    # t_pcg2_2 = time.perf_counter()

    # print(f"normal used: {t_normal_2-t_normal_1}")
    print(f"Jacobi used: {t_Jacobi_2 - t_Jacobi_1}")
    print(f"G-S used: {t_GS_2 - t_GS_1}")
    print(f"SOR used: {t_SOR_2 - t_SOR_1}")
    # print(f"cg used: {t_cg_2-t_cg_1}")
    # print(f"pcg1 used: {t_pcg1_2-t_pcg1_1}")
    # print(f"pcg2 used: {t_pcg2_2-t_pcg2_1}")

if __name__ == '__main__':
    main()
