from sklearn.feature_extraction import DictVectorizer   # 结构数据转成整数
'''sklearn要求被处理的数据必须是数值型的，不能是字符串类型的，例如，对于age(youth,senior,middle_age)
可以拆解为：
    youth 1
    middle_age 0
    senior 0
三个属性，值为1则证明age是这个属性，否则证明age不是这个属性
'''
import csv                                              # 读取csv文件，自带
from sklearn import preprocessing                       # 预处理
from sklearn import tree                                # 决策树
from sklearn.externals.six import StringIO              # 文件读写
from astropy.wcs.docstrings import row

# import training data
    # open file
allElectronicsData = open(r"../DecisionTreeDataSets.csv", "r")
    # read by line
reader = csv.reader(allElectronicsData)
    # read header
header = next(reader)
# print(header)

# read features and corresponding labels
featureList = []    # no or yes
labelList = []      # youth or not, middle_age or not...
for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    # get rid of RID and feature
    for prosperity in range(1, len(row) - 1):
        rowDict[header[prosperity]] = row[prosperity]
    featureList.append(rowDict)
# print(labelList)
# print(featureList)

# translate features of dict array format to integer array
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList, None).toarray()
# print(dummyX)

# translate labels of string array format to interger array
labels = preprocessing.LabelBinarizer()
dummyY = labels.fit_transform(labelList, None)
# print(dummyY)

# preprocessing terminated, we use DecisionTreeClassifier function directly
clf = tree.DecisionTreeClassifier(criterion='entropy')  # entropy:熵
clf = clf.fit(dummyX, dummyY)
# print(str(clf))

# visualization
    # with sentence make sure the file handle will be closed even if some exceptions thrown
with open(r"allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names = vec.get_feature_names(), out_file = f)
    
# prediction
newX = dummyX[0, :]
newX[0] = 1
newX[2] = 0

predictY = clf.predict(newX)
print(newX,':',predictY)