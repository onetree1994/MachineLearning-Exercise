import numpy as np
import math

# 计算 Pearson Correlation Coefficient
def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    
    SST = math.sqrt(varX * varY)
    return SSR / SST

def polyfit(x, y, degree = 1):
    results = {}
    # 计算回归方程系数，degree means most high orders(最高次幂)
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()
    p = np.poly1d(coeffs) # 计算预测值
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot
    return results

testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]
print('Pearson Correlation Coefficient: ', computeCorrelation(testX, testY))

print('R^2: ', polyfit(testX, testY))