import numpy as np
from pydub import AudioSegment
import os
import random

def loader(path, testingAmount = 0.8):
    megaList = []
    X = []
    Y = []
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            sound = AudioSegment.from_wav(path)
            #X.append(sound)
            label = root.split("\\")
            x = sound.get_array_of_samples()
            x = np.array(x)
            megaList.append((x, label[len(label)-1]))
            #Y.append(label[len(label)-1])
           # print(label[len(label)-1])
    random.shuffle(megaList)
    for x,y in megaList:
        X.append(x)
        Y.append(fromStringToInt(y))
    x_Train = np.array(X[:int(X.__len__()*testingAmount)])
    y_Train = np.array(Y[:int(Y.__len__()*testingAmount)])
    x_Test = np.array(X[x_Train.__len__():])
    y_Test = np.array(Y[y_Train.__len__():])
    return(x_Train,y_Train),(x_Test,y_Test)

def fromStringToInt(string):
    if string == "CD_quality":
        return 0
    else:
        return 1

def fromIntToString(int):
    if int == 0:
        return "CD_quality"
    else:
        return "LP_quality"
loader("training")