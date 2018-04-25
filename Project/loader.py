import numpy as np
#import audiosegment
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def loaderPictures(path, testingAmount = 0.8, row =28, col=28, shuffle=False):
    megaList = []
    X = []
    Y = []
    c = 0
    #data = np.empty((2000, col, row), dtype=np.uint8)
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if not file.__contains__('db'):
                label = root.split("/")
                with Image.open(path) as f:
                    megaList.append((((np.array(f.convert('RGB').resize((row, col))))/255), label[len(label)-1]))
                c+=1
    if shuffle:
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

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray