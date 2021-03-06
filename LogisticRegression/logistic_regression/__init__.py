import numpy as np
import random

# 梯度下降法, 注意没有使用sigmoid函数！！
def gradientDesent(x, y, theta, alpha, m, numIterater):
    xTrans = x.transpose()              # 转置
    for i in range(numIterater):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        print("cost:", cost)
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta
# 数据生成函数
def genData(numPoints, bias, variance):
    x = np.zeros(shape = (numPoints, 2))
    y = np.zeros(shape = numPoints)
    # straight line
    for i in range(0, numPoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

x, y = genData(100, 25, 10)
m, n = np.shape(x)

print("theta:", gradientDesent(x, y, np.ones(n), 0.0005, m, numIterater = 100000))