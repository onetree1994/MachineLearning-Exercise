'''from sklearn import neighbors
from sklearn import datasets

# import classifier
knn = neighbors.KNeighborsClassifier()

# import iris datasets
iris = datasets.load_iris()
# print(iris)

knn.fit(iris.data, iris.target)
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4], [0.4, 0.2, 0.3, 0.4]])
print(predictedLabel)'''

# imply KNN directly
import csv
import random
import math
import operator
from audioop import reverse
from tornado.test.routing_test import GetResource
from nltk.chunk.util import accuracy

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
#         print(dataset)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
     
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += math.pow(instance1[x]-instance2[x], 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes)
#     print(classVotes)
    return sortedVotes[0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
                
if __name__ == "__main__":
    trainingSet = []
    testSet = []
#     print(trainingSet)
    loadDataset(r"../iris.data.txt", 0.5, trainingSet, testSet)
#     print(trainingSet)
#     print(testSet)
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
#         print(result)
        predictions.append(result)
    print(predictions)
    print(getAccuracy(testSet, predictions))