import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


DNA_SIZE = 26  # 个体编码长度
POPULATION_SIZE = 200  # 种群大小
GENERATION_NUMBER = 200  # 世代数目
CROSS_RATE = 0.8  # 交叉率
VARIATION_RATE = 0.01  # 变异率
X_RANGE = [-32.768, 32.768]  # X范围
Y_RANGE = [-32.768, 32.768]  # Y范围


# 问题函数 阿克莱
# @param x x坐标
# @param y y坐标
# @return z 函数值
def ackleyFunc(x, y):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = 2
    sum1 = x * x + y * y
    sum2 = np.cos(c * x) + np.cos(c * y)
    term1 = -a * np.exp(-b * np.sqrt(1 / d * sum1))
    term2 = -np.exp(1 / d * sum2)
    z = term1 + term2 + a + np.exp(1)
    return z


# 初始化图
# @param ax 3D图像
def init3DGraph(ax):
    x_sequence = np.linspace(*X_RANGE, 100)  # 创建x等差数列
    y_sequence = np.linspace(*Y_RANGE, 100)  # 创建y等差数列
    x_matrix, y_matrix = np.meshgrid(x_sequence, y_sequence)  # 生成x和y的坐标矩阵
    z_matrix = ackleyFunc(x_matrix, y_matrix)  # 生成z坐标矩阵
    # 创建曲面图,行跨度为1，列跨度为1，设置颜色映射
    ax.plot_surface(x_matrix, y_matrix, z_matrix, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'))
    ax.set_zlim(0, 25)  # 自定义z轴范围
    ax.set_xlabel('x')  # 设置x坐标轴标题
    ax.set_ylabel('y')  # 设置y坐标轴标题
    ax.set_zlabel('z')  # 设置z坐标轴标题
    plt.pause(2)  # 暂停2秒
    plt.show()  # 显示图


# 解码DNA个体
# @param population_matrix 种群矩阵
# @return population_x_vector, population_y_vector 种群x向量，种群y向量
def decodingDNA(population_matrix):
    x_matrix = population_matrix[:, 1::2]  # 矩阵分割，行不变，抽取奇数列作为x矩阵
    y_matrix = population_matrix[:, 0::2]  # 矩阵分割，行不变，抽取偶数列作为y矩阵
    # 解码向量，用于二进制转十进制，其值为[2^23 2^22 ... 2^1 2^0]，对位相乘累加，二进制转十进制的基础方法
    decoding_vector = 2 ** np.arange(DNA_SIZE)[::-1]
    # 种群x向量，由二进制转换成十进制并映射到x区间
    population_x_vector = x_matrix.dot(decoding_vector) / (2 ** DNA_SIZE - 1)\
                          * (X_RANGE[1] - X_RANGE[0]) + X_RANGE[0]
    # 种群y向量，由二进制转换成十进制并映射到y区间
    population_y_vector = y_matrix.dot(decoding_vector) / (2 ** DNA_SIZE - 1)\
                          * (Y_RANGE[1] - Y_RANGE[0]) + Y_RANGE[0]
    return population_x_vector, population_y_vector


# DNA交叉
# @param child_DNA 孩子DNA
# @param population_matrix 种群矩阵
def DNACross(child_DNA, population_matrix):
    # 概率发生DNA交叉
    if np.random.rand() < CROSS_RATE:
        mother_DNA = population_matrix[np.random.randint(POPULATION_SIZE)]  # 种群中随机选择一个个体作为母亲
        cross_position = np.random.randint(DNA_SIZE * 2)  # 随机选取交叉位置
        child_DNA[cross_position] = mother_DNA[cross_position]  # 孩子获得交叉位置处母亲基因


# DNA变异
# @param child_DNA 孩子DNA
def DNAVariation(child_DNA):
    # 概率发生DNA变异
    if np.random.rand() < VARIATION_RATE:
        variation_position = np.random.randint(DNA_SIZE * 2)  # 随机选取变异位置
        child_DNA[variation_position] = child_DNA[variation_position] ^ 1  # 异或门反转二进制位


# 更新种群
# @param population_matrix 种群矩阵
# @return new_population_matrix 更新后的种群矩阵
def updatePopulation(population_matrix):
    new_population_matrix = []  # 声明新的空种群
    # 遍历种群所有个体
    for father_DNA in population_matrix:
        child_DNA = father_DNA  # 孩子先得到父亲的全部DNA（染色体）
        DNACross(child_DNA, population_matrix)  # DNA交叉
        DNAVariation(child_DNA)  # DNA变异
        new_population_matrix.append(child_DNA)  # 添加到新种群中
    new_population_matrix = np.array(new_population_matrix)  # 转化数组
    return new_population_matrix


# 获取适应度向量
# @param population_matrix 种群矩阵
# @return fitness_vector 适应度向量
def getFitnessVector(population_matrix):
    population_x_vector, population_y_vector = decodingDNA(population_matrix)  # 获取种群x和y向量
    fitness_vector = ackleyFunc(population_x_vector, population_y_vector)  # 获取适应度向量
    # print(fitness_vector)
    fitness_vector = np.max(fitness_vector) - fitness_vector + 1e-3 # 保证适应度大于0 并且原适应度越小 得到的值越大 #fitness_vector - np.min(fitness_vector) + 1e-3  # 适应度修正，保证适应度大于0
    return fitness_vector


# 自然选择
# @param population_matrix 种群矩阵
# @param fitness_vector 适应度向量
# @return population_matrix[index_array] 选择后的种群
def naturalSelection(population_matrix, fitness_vector):
    index_array = np.random.choice(np.arange(POPULATION_SIZE),  # 被选取的索引数组
                                   size=POPULATION_SIZE,  # 选取数量
                                   replace=True,  # 允许重复选取
                                   p=fitness_vector / fitness_vector.sum())  # 数组每个元素的获取概率
    # print(population_matrix[index_array])
    return population_matrix[index_array]

# 获取当前迭代最优解
def getOptSol(population_matrix):
    fitness_vector = getFitnessVector(population_matrix)  # 获取适应度向量
    optimal_fitness_index = np.argmax(fitness_vector)#np.argmax(fitness_vector)  # 获取最大适应度索引

    return ackleyFunc(population_x_vector[optimal_fitness_index], population_y_vector[optimal_fitness_index])

# 打印结果
# @param population_matrix 种群矩阵
def printResult(population_matrix):
    fitness_vector = getFitnessVector(population_matrix)  # 获取适应度向量
    optimal_fitness_index = np.argmax(fitness_vector)#np.argmax(fitness_vector)  # 获取最大适应度索引
    print('最佳适应度为: ', fitness_vector[optimal_fitness_index])
    print('最优基因型为: ', population_matrix[optimal_fitness_index])
    population_x_vector, population_y_vector = decodingDNA(population_matrix)  # 获取种群x和y向量
    print('最优基因型十进制表示为: ',
          (population_x_vector[optimal_fitness_index], population_y_vector[optimal_fitness_index]))
    print('最优函数值为: ',
          ackleyFunc(population_x_vector[optimal_fitness_index], population_y_vector[optimal_fitness_index]))



if __name__ == '__main__':
    fig = plt.figure()  # 创建空图像
    ax = Axes3D(fig)  # 创建3D图像
    plt.ion()  # 切换到交互模式绘制动态图像
    init3DGraph(ax)  # 初始化图
    # 生成随机种群矩阵，这里DNA_SIZE * 2是因为种群矩阵要拆分为x和y矩阵，单条DNA（染色体、个体）长度为24
    # 若视x和y为等位基因，x和y组成染色体对，共同影响个体，这里巧妙地与遗传信息对应起来
    population_matrix = np.random.randint(2, size=(POPULATION_SIZE, DNA_SIZE * 2))

    #画折线图
    # plt.axis([0, 100, 0, 1])
    # plt.ion()
    optSol = [] #最优解
    iter = [] #迭代次数
    # 迭代50世代
    for i in range(GENERATION_NUMBER):
        population_x_vector, population_y_vector = decodingDNA(population_matrix)  # 获取种群x和y向量
        # 绘制散点图，设置颜色和标记风格
        ax.scatter(population_x_vector,
                   population_y_vector,
                   ackleyFunc(population_x_vector, population_y_vector),
                   c='b',
                   marker='*')
        plt.show()  # 显示图
        plt.pause(0.01)  # 暂停0.1秒
        #绘制迭代-最优解图
        iter.append(i)
        optSol.append(getOptSol(population_matrix))
        # print(optSol)
        # print(iter)
        plt.figure(2)
        plt.plot(iter, optSol)
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.xlabel('迭代次数')
        plt.ylabel('最优解')
        plt.show()
        #
        population_matrix = updatePopulation(population_matrix)  # 更新种群
        fitness_vector = getFitnessVector(population_matrix)  # 获取适应度向量
        population_matrix = naturalSelection(population_matrix, fitness_vector)  # 自然选择

        if(i==GENERATION_NUMBER-1):
            iter.append(i+1)
            optSol.append(getOptSol(population_matrix))

    # print(optSol)
    # print(iter)
    # plt.figure(2)
    # plt.plot(iter, optSol)
    # plt.xlabel('Group')
    # plt.ylabel('Count')
    # plt.show()

    printResult(population_matrix)  # 打印结果
