from __future__ import print_function
from time import time
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing.tests.test_data import n_features

# 自行添加的包
import matplotlib.image as mpimg
import os
import numpy as np
from implement import NeuralNetwork
from sklearn import decomposition

pic1path = r'.././dataset/0'
pic1 = []
pic1label = []
pic2path = r'.././dataset/1'
pic2 = []
pic2label = []
samplepath = r'.././dataset/sample'
sample = []
# rgb转为灰度图
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# 抽取图片数据为行向量，合成样本
print("Reading training dataset...")
for filename in os.listdir(pic1path):
    filepath = pic1path + r'/' + filename
    img = rgb2gray(mpimg.imread(filepath))
    limg = []
    for line in img:
        limg.extend(line)
    pic1.append(limg)
    pic1label.extend([0])
for filename in os.listdir(pic2path):
    filepath = pic2path + r'/' + filename
    img = rgb2gray(mpimg.imread(filepath))
    limg = []
    for line in img:
        limg.extend(line)
    pic2.append(limg)
    pic2label.extend([1])
# 合成总样本X和标签y
X = []
X.extend(pic1)
X.extend(pic2)
y = []
y.extend(pic1label)
y.extend(pic2label)
print("\nReading training dataset finished~")
# 此处应该使用PCA降维提取特征值，具体实现以后仔细研究
# pca = PCA(svd_solver='randomized').fit(X)
print("PCAing to reduce the dimension...")
pca = decomposition.PCA()
nX = pca.fit_transform(X)
print(nX)
X = nX
print("\nPCA complete~")
# 抽取测试样本，采用最邻近分类法
print("\nReading test dataset...")
for filename in os.listdir(samplepath):
    filepath = samplepath + r'/' + filename
    img = rgb2gray(mpimg.imread(filepath))
    limg = []
    for line in img:
        limg.extend(line)
    sample = limg
print("\nReading test dataset finished~")
# 暴力分类--最近值分类
# sampleclass = 0
# mindis = 0
# for row in range(len(X) - 1):
#     addsum = 0
#     for i in range(len(X[row]) - 1):
#         addsum = addsum + pow((X[row][i] - sample[i]), 2)
#     if((addsum < mindis) & (mindis != 0.0)):
#         mindis = addsum
#         print(addsum)
#         sampleclass = y[row]
#     elif(mindis == 0):
#         mindis = addsum
#         print(addsum)
#         sampleclass = y[row]
# print("The number is:", sampleclass)

# 利用神经网络进行数字识别
nn = NeuralNetwork([len(X[0]), 10, 5, 2, 1])
print("\nBegin training with NeuralNetwork...")
nn.fit(X, y, epochs = 10000)
print("\nTraining over~")
test = []
test.append(sample)
test = pca.transform(test)
print(nn.predict(test[0]))