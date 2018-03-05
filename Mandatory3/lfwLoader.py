import os
import scipy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
def loader(testingAmount = 0.8):
    X = []
    Y = []
    for root, dirs, files in os.walk('lfw'):
        for file in files:
            split = file.split('.')
            if not (split[1].__contains__('db')):
                im = Image.open(os.path.join(root,file)).convert('LA')
                pic = np.array(im)
                pic = pic/255
                X.append(pic)
                name = file.split('_')
                Y.append(name[0]+name[1])
    x_Train = np.array(X[:int(X.__len__()*testingAmount)])
    y_Train = np.array(Y[:int(Y.__len__()*testingAmount)])
    x_Test = np.array(X[x_Train.__len__():])
    y_Test = np.array(Y[y_Train.__len__():])
    return(x_Train,y_Train),(x_Test,y_Test)
