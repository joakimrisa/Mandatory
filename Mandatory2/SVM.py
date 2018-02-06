import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import os
import warnings

warnings.simplefilter("ignore")


def loader():
    org = 'signs'
    allData = dict()

    for root, dirs, files in os.walk('signs', topdown=False):
        #print(root)
        for file in files:
            newFile = file.split('.')
            newFile = newFile[0][:newFile[0].__len__()-1]
            if not newFile in allData:
                allData[newFile] = []
            dataList = []

            appendData = [[i for i in i.split(",")] for i in open(os.path.join(root, file)).readlines() if i.strip()]
            for d in appendData:
                floatList = []
                counter = 0
                for num in d:
                    if counter == len(d)-4:
                        break
                    floatList.append(float(num))
                    counter += 1
                dataList.append(floatList)
            allData[newFile] += (dataList)
        #print(allData['alive'])
            #print(dataList)
    trainingData = dict()
    testData = dict()
    for key in allData:
        trainingNum = int(allData[key].__len__() * 0.8)
        #testNum = allData[key].__len__() - trainingNum
        trainingData[key] = allData[key][:trainingNum]
        testData[key] = allData[key][trainingNum:]
    return trainingData, testData

    #print(trainingDict.keys())
    #print(trainingDict)
        #with open(os.path.join(org,files)) as f:
            #   print(f)

trainingData, testData = loader()

'''
training = [[int(i) for i in i.split(",")] for i in open("pendigits.tra").readlines() if i.strip()]
testing = [[int(i) for i in i.split(",")] for i in open("pendigits.tes").readlines() if i.strip()]

training_0 = [i[:-1] for i in training if i[-1] == 0]
training_1 = [i[:-1] for i in training if i[-1] == 1]

testing_0 = [i[:-1] for i in testing if i[-1] == 0]
testing_1 = [i[:-1] for i in testing if i[-1] == 1]


# Mapping to 2d --- for plot purposes
'''
def mapTo2D(data):
    retval = []
    for i in range(0, len(data)-1, 2):
        x = data[i]
        y = data[i + 1]
        retval.append((x, y))
    return retval

trainingMap2D = dict()
for key in trainingData:
    for d in trainingData[key]:
        if key not in trainingMap2D:
            trainingMap2D[key] = []
        trainingMap2D[key] += (mapTo2D(d))
testMap2D = dict()
for key in testData:
    for d in testData[key]:
        if key not in testMap2D:
            testMap2D[key] = []
        testMap2D[key] += (mapTo2D(d))

#print(trainingMap2D)

def returnXY(data, n=8):
    X = []
    Y = []
    for key in data:
        X += data[key][:n]
        Y += [key for i in data[key][:n]]
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

X, Y = returnXY(trainingData)


C = 1.0
gamma = 0.5


svm_linear = svm.SVC(kernel='linear', C=C, gamma=gamma).fit(X, Y)
svm_rbf = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(X, Y)
svm_sigmoid = svm.SVC(kernel='sigmoid', C=C, gamma=gamma).fit(X, Y)
svm_poly = svm.SVC(kernel='poly', C=C, gamma=gamma).fit(X, Y)

r = svm_poly.predict(testData['hurt'][:8])
print(r)
r = svm_rbf.predict(testData['hurt'][:8])
print(r)
r = svm_sigmoid.predict(testData['hurt'][:8])
print(r)
def testSVM(svm, testDict):
    numcorrect = 0.
    numwrong = 0.
    answerDict = dict()
    for key in testDict:
        r = svm.predict(testDict[key])
        for n in r:
            if n == key:
                numcorrect += 1
            else:
                numwrong += 1
        answerDict[key] = (numcorrect, numwrong)
        numcorrect = 0
        numwrong = 0

    return answerDict
#print(testMap2D['alive'])
print(trainingData['alive'].__len__())
print(trainingData['alive'])
print(testData['alive'].__len__())
print("Linear", testSVM(svm_linear, testData))

print("RBF", testSVM(svm_rbf, testData))

print("Sigmoid", testSVM(svm_sigmoid, testData))


'''
training_2d_0 = []
training_2d_1 = []

for d in training_0:
    training_2d_0 += mapTo2D(d)

for d in training_1:
    training_2d_1 += mapTo2D(d)

# Plotting 2d
plt.subplot(2, 2, 1)
plt.plot([i[0] for i in training_2d_0], [i[1] for i in training_2d_0], "-o", color="green")

plt.subplot(2, 2, 2)
plt.plot([i[0] for i in training_2d_1], [i[1] for i in training_2d_1], "-o", color="green")

plt.subplot(2, 2, 3)
plt.plot([i[0] for i in training_2d_0][:8], [i[1] for i in training_2d_0][:8], "-o", color="green")

plt.subplot(2, 2, 4)
plt.plot([i[0] for i in training_2d_1][:8], [i[1] for i in training_2d_1][:8], "-o", color="green")
plt.show()

X = np.array(training_2d_0[:8] + training_2d_1[:8])
Y = np.array([0 for i in training_2d_0[:8]] + [1 for i in training_2d_1[:8]])

C = 1.0
gamma = 0.5
svm_linear = svm.SVC(kernel='linear', C=C, gamma=gamma).fit(X, Y)
svm_rbf = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(X, Y)
svm_sigmoid = svm.SVC(kernel='sigmoid', C=C, gamma=gamma).fit(X, Y)

h = 0.2  # Mesh step
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


def plotSVM(svm, n, title):
    plt.subplot(2, 2, n)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.title(title)


plotSVM(svm_linear, 1, "Linear")
plotSVM(svm_rbf, 2, "RBF")
plotSVM(svm_sigmoid, 3, "Sigmoid")

plt.show()

testing_2d_0 = []
testing_2d_1 = []
for d in testing_0:
    testing_2d_0 += mapTo2D(d)

for d in testing_1:
    testing_2d_1 += mapTo2D(d)


def testSVM(svm, zero, one):
    numcorrect = 0.
    numwrong = 0.
    for correct, testing in ((0, zero), (1, one)):
        for d in testing:
            r = svm.predict(d)[0]
            if (r == correct):
                numcorrect += 1
            else:
                numwrong += 1
    print("Correct", numcorrect)
    print("Wrong", numwrong)
    print("Accuracy", (numcorrect) / (numcorrect + numwrong))
'''
'''
print("Linear",testSVM(svm_linear, testing_2d_0, testing_2d_1))

print("RBF",testSVM(svm_rbf, testing_2d_0, testing_2d_1))

print("Sigmoid",testSVM(svm_sigmoid, testing_2d_0, testing_2d_1))

# 16d data
X = np.array(training_0 + training_1)
Y = np.array([0 for i in training_0] + [1 for i in training_1])

svm_linear = svm.SVC(kernel='linear', C=C, gamma=gamma).fit(X, Y)
svm_poly = svm.SVC(kernel='poly', C=C, gamma=gamma).fit(X, Y)
svm_rbf = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(X, Y)
svm_sigmoid = svm.SVC(kernel='sigmoid', C=C, gamma=gamma).fit(X, Y)

print
"Linear"
testSVM(svm_linear, testing_0, testing_1)

print
"Polinomial"
testSVM(svm_poly, testing_0, testing_1)

print
"RBF"
testSVM(svm_rbf, testing_0, testing_1)

print
"Sigmoid"
testSVM(svm_sigmoid, testing_0, testing_1)
'''