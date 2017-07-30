from implement import NeuralNetwork
from sklearn.datasets import load_digits
import pylab as pl
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing.label import LabelBinarizer
from sklearn.utils.estimator_checks import check_clusterer_compute_labels_predict
from sklearn.metrics.classification import classification_report
from sklearn import decomposition

# 每张都是64像素的灰度图
digits = load_digits()
print(digits.data.shape)
X = digits.data
y = digits.target
X -= X.min() # 标准化
X /= X.max()

nn = NeuralNetwork([64, 100, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)

print("start dimensionality reduction......")
pca = decomposition.PCA()
newX = pca.fit_transform(X_train)
print(newX)
X_train = newX
print(X_train.shape)
print("dimensionality reduction over, but pca seems not useful in this kind of dataset...")

print("start fitting......")
nn.fit(X_train, labels_train, epochs = 100000)
print("fitting over~")

predictions = []
print("\nstart comparing......")
for i in range(X_test.shape[0]):
    ins = []
    ins.append(X_test[i])
    ins = pca.transform(ins)
    o = nn.predict(ins[0])
    predictions.append(np.argmax(o)) # 整化
print(confusion_matrix(y_test, predictions))
print("\n\n", classification_report(y_test, predictions))