import numpy as np
from sklearn import svm
import os
import warnings
import itertools

warnings.simplefilter("ignore")


def loader():
    '''
    This function loads in the necessary data
    '''
    org = 'signs'
    allData = dict()

    for root, dirs, files in os.walk('signs', topdown=False):
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
                dataList += (floatList)
            allData[newFile].append(dataList)
    trainingData = dict()
    testData = dict()
    for key in allData:
        trainingNum = int(allData[key].__len__() * 0.8)
        trainingData[key] = allData[key][:trainingNum]
        testData[key] = allData[key][trainingNum:]
    return trainingData, testData

trainingData, testData = loader()

def mapTo2D(data):
    '''
    This function maps the data into 2D
    '''
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


def returnXY(data, attriSet, n=8):
    '''
    This function takes in data and attribute set and n.
    Returns X and Y
    '''
    X = np.empty((data.__len__()*n, len(attriSet), 60))
    Y = []
    c = 0
    c2 = 0
    c3 = 0
    for key in data:
        for k in data[key][:n]:
            for v in k:
                if attriSet.__contains__(c2):
                    X[c][c2][c3] = v

                c2 += 1
                if c2 == max(attriSet):
                    c3 += 1
                    c2 = 0
                    if c3 == 60:
                        c += 1
                        c3 = 0
                        break

        Y += [key for i in data[key][:n]]
    Y = np.array(Y)
    return X, Y

def permSet():
    '''
    This function calculates the combinations
    '''
    possibleNumbras = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    permas = set()
    for n in range(11):
        for perum in itertools.combinations(possibleNumbras, n+1):
            permas.add(perum)
    return permas


def runForever2():
    '''
    This function is the runner for the task
    '''
    bestDict = dict()
    bestDict['linear'] = 0
    bestDict['rbf'] = 0
    bestDict['sigmoid'] = 0
    bestDict['poly'] = 0
    X, Y = returnXY(trainingData, [0,1,2,3], n=25)
    X = X.reshape(X.shape[0], -1)
    testX, testY = returnXY(testData, [0,1,2,3], n=12)
    testX = testX.reshape(testX.shape[0], -1)
    cRange = np.arange(0.1, 20, 0.2)
    gRange = np.arange(0.1, 20, 0.1)
    for C in cRange:
        for gamma in gRange:
            svm_linear = svm.SVC(kernel='linear', C=C, gamma=gamma).fit(X, Y)
            svm_rbf = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(X, Y)
            svm_sigmoid = svm.SVC(kernel='sigmoid', C=C, gamma=gamma).fit(X, Y)
            svm_poly = svm.SVC(kernel='poly', C=C, gamma=gamma).fit(X, Y)
            correct, _ = checkSVM(svm_linear, testX, testY)
            if correct > bestDict['linear']:
                print('linear: ', correct, ' C: ', C, ' Gamma: ', gamma)
                bestDict['linear'] = correct

            correct, _ = checkSVM(svm_rbf, testX, testY)
            if correct > bestDict['rbf']:
                print('rbf: ', correct, ' C: ', C, ' Gamma: ', gamma)
                bestDict['rbf'] = correct

            correct, _ = checkSVM(svm_sigmoid, testX, testY)
            if correct > bestDict['sigmoid']:
                print('sigmoid: ', correct, ' C: ', C, ' Gamma: ', gamma)
                bestDict['sigmoid'] = correct

            correct, _ = checkSVM(svm_poly, testX, testY)
            if correct > bestDict['poly']:
                print('poly: ', correct, ' C: ', C, ' Gamma: ', gamma)
                bestDict['poly'] = correct

def runForever():
    '''
    This function is the first runner for the task. This runner considered the combinations of features as well
    '''
    combinations = permSet()
    bestDict = dict()
    bestDict['linear'] = 0
    bestDict['rbf'] = 0
    bestDict['sigmoid'] = 0
    bestDict['poly'] = 0
    for combo in combinations:
        X, Y = returnXY(trainingData, combo, n=25)
        X = X.reshape(X.shape[0], -1)
        testX, testY = returnXY(testData, combo, n=12)
        testX = testX.reshape(testX.shape[0], -1)
        cRange = np.arange(0.1, 20, 0.2)
        gRange = np.arange(0.1, 20, 0.1)
        for C in cRange:
            for gamma in gRange:
                svm_linear = svm.SVC(kernel='linear', C=C, gamma=gamma).fit(X, Y)
                svm_rbf = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(X, Y)
                svm_sigmoid = svm.SVC(kernel='sigmoid', C=C, gamma=gamma).fit(X, Y)
                svm_poly = svm.SVC(kernel='poly', C=C, gamma=gamma).fit(X, Y)
                correct, _ = checkSVM(svm_linear, testX, testY)
                if correct > bestDict['linear']:
                    print('linear: ', correct, ' C: ', C, ' Gamma: ', gamma, ' Combo: ', combo)
                    bestDict['linear'] = correct

                correct, _ = checkSVM(svm_rbf, testX, testY)
                if correct > bestDict['rbf']:
                    print('rbf: ', correct, ' C: ', C, ' Gamma: ', gamma, ' Combo: ', combo)
                    bestDict['rbf'] = correct

                correct, _ = checkSVM(svm_sigmoid, testX, testY)
                if correct > bestDict['sigmoid']:
                    print('sigmoid: ', correct, ' C: ', C, ' Gamma: ', gamma, ' Combo: ', combo)
                    bestDict['sigmoid'] = correct

                correct, _ = checkSVM(svm_poly, testX, testY)
                if correct > bestDict['poly']:
                    print('poly: ', correct, ' C: ', C, ' Gamma: ', gamma, ' Combo: ', combo)
                    bestDict['poly'] = correct

def checkSVM(svm, X, Y):
    '''
    This function checks the SVM and returns the number of correct and wrong
    '''
    r = svm.predict(X)
    correct = 0
    wrong = 0
    for n in range(len(Y)):
        if r[n] == Y[n]:
            correct += 1
        else:
            wrong += 1
    return correct, wrong

def testSVM(svm, testDict):
    '''
    This function checks the SVM and returns a dict of correct and wrong.
    Used to test accuracy of multiple SVM in one run
    '''
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

runForever2()