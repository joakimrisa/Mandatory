import os
import scipy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import translate
def loader(testingAmount = 0.8):
    X = []
    Y = []

    for root, dirs, files in os.walk('gender'):
        for file in files:
            split = file.split('.')
            basewidth = 64
            if not (split[1].__contains__('db')):
                im = Image.open(os.path.join(root,file))
                wpercent = (basewidth / float(im.size[0]))
                hsize = int((float(im.size[1]) * float(wpercent)))
                im = im.resize((basewidth, hsize), Image.ANTIALIAS)
                #im.show()
                pic = np.array(im)
                pic = rgb2gray(pic)
                pic = pic/255
                #print(pic.shape)
                X.append(pic)
                name = file.split('_')
                #Y.append(name[0]+name[1])
                Y.append(translate.toNumber(name[2]))
                #print(name[2])
    x_Train = np.array(X[:int(X.__len__()*testingAmount)])
    y_Train = np.array(Y[:int(Y.__len__()*testingAmount)])
    x_Test = np.array(X[x_Train.__len__():])
    y_Test = np.array(Y[y_Train.__len__():])

    return(x_Train,y_Train),(x_Test,y_Test)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray