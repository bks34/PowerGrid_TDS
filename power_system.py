"""
PowerSystem电网模型
稀疏矩阵
"""
import numpy as np
from scipy.sparse import csr_array, csc_array, dia_array

class PowerSystem:
    """
    用于生成网格
    """
    def __init__(self):
        self.node_size = 0
        # 矩阵B为稀疏矩阵
        self.B = None

        # M,D,T,X都为对角矩阵
        self.M = None
        self.D = None
        self.T = None
        self.X = None

        # P,E^{f}是列向量
        self.P = None
        self.Ef = None

        # 要求的变量
        self.w = None
        self.theta = None
        self.E = None


    def read_from_psat_file(self, filename: str):
        """
        仅从psat的案例中读取网络的拓扑结构
        :param filename: psat的tests文件夹中的mdl.m文件
        :return: self
        """
        matrices = self.__parse_matlab_matrix(filename)

        # 需要使用的矩阵
        bus_con = matrices['Bus.con']
        line_con = matrices['Line.con']

        # 规模
        self.node_size = bus_con.shape[0]

        # 固定随机种子
        np.random.seed(42)

        # 计算B
        self.B = np.zeros((self.node_size, self.node_size))
        for line in line_con:
            i = int(line[0]-1)
            j = int(line[1]-1)
            self.B[i][j] = -np.random.uniform(8, 12)
            self.B[j][i] = self.B[i][j]
        for i in range(self.node_size):
            self.B[i][i] = np.sum(-self.B[i,:])
        self.B = csc_array(self.B) #转为稀疏矩阵

        self.__generate_others()

        return self

    def read_from_matpower_file(self, filename: str):
        """
        仅从matpower的案例中读取网络的拓扑结构
        :param filename: matpower的data文件夹中的文件
        :return: self
        """
        matrices = self.__parse_matlab_matrix(filename)

        # 需要使用的矩阵
        mpc_bus = matrices['mpc.bus']
        mpc_branch = matrices['mpc.branch']

        # 规模
        self.node_size = mpc_bus.shape[0]

        # 固定随机种子
        np.random.seed(42)

        # 计算B
        self.B = np.zeros((self.node_size, self.node_size))
        for line in mpc_branch:
            i = int(line[0] - 1)
            j = int(line[1] - 1)
            self.B[i][j] = np.random.uniform(8, 12)
            self.B[j][i] = self.B[i][j]
        for i in range(self.node_size):
            self.B[i][i] = np.sum(-self.B[i, :])
        self.B = csc_array(self.B) #转为稀疏矩阵

        self.__generate_others()

        return self


    @staticmethod
    def __parse_matlab_matrix(filename: str):
        """
        :param filename: psat的tests文件夹中的mdl.m文件
        :return: 包含所有matrix的hash表(字典)
        """
        matrices = {}
        current_mat_name = None
        current_mat_data = []

        with open(filename,"r") as file:
            for line in file:
                line = line.strip()

                # 跳过空行与注释行
                if not line or line.startswith("%"):
                    continue

                # 矩阵开头
                if line.find('=') >=0 and line.find('[') >=0:
                    current_mat_name = line.split('=')[0].strip()
                    current_mat_data = []
                    continue

                # 矩阵结尾
                if line.find('];') >= 0:
                    matrices[current_mat_name] = np.array(current_mat_data)
                    current_mat_name = None
                    continue

                # 处理数据行
                if current_mat_name is not None:
                    tail_index = line.find(';')
                    elements = line[0:tail_index].split()
                    row = [float(x) for x in elements[0:2]] # 暂时只用前两列
                    current_mat_data.append(row)

        return matrices

    def __generate_others(self):
        """
        随机生成惯性M，注入功率P，阻尼D，初始条件w和theta
        :return:
        """
        # 惯性M
        self.M = dia_array(np.diag(np.random.uniform(1, 4, size=self.node_size)))
        # 阻尼
        self.D = dia_array(np.diag(np.random.uniform(2, 5, size=self.node_size)))
        # T
        self.T = dia_array(np.diag(np.random.uniform(0.2,1, size=self.node_size)))
        # X
        self.X = dia_array(np.diag(np.random.uniform(0.001, 0.01, size=self.node_size)))


        # 注入功率
        self.P = np.random.uniform(-0.5, 0.5, size=self.node_size).reshape(self.node_size, 1)
        self.P -= np.mean(self.P)  # 确保功率平衡

        # E^{f}
        self.Ef = np.random.uniform(0.9,1.1,size=self.node_size).reshape(self.node_size, 1)


        # 初始条件（小扰动）
        self.w = np.random.normal(scale=0.2, size=self.node_size).reshape(self.node_size, 1)  # 初始角速度
        self.theta = np.random.normal(scale=0.2, size=self.node_size).reshape(self.node_size, 1)  # 初始角度
        self.E = np.abs(np.random.normal(scale=0.2, size=self.node_size).reshape(self.node_size, 1))   #初始电压
