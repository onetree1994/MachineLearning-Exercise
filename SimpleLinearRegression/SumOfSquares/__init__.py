import numpy as np

def fitSLR(x, y):
    """简单线性回归
    :param x: 样本x坐标
    :param y: 样本y坐标
    x和y必须是同长度的
    """
    n = len(x)
    denominator = 0 # 分母
    numerator = 0   # 分子
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    for i in range(n):
        numerator += (x[i] - x_mean)*(y[i] - y_mean)
        denominator += (x[i] - x_mean)**2
    
    b1 = numerator / float(denominator)
    b0 = y_mean - b1*x_mean
    print(b1,'x +',b0,'= y')
    return b0,b1

def predict(x, b0, b1):
    return b0 + b1*x

x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]

b = fitSLR(x, y)

print(predict(6, b[0], b[1]))