import numpy as np
import audiosegment
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def loaderPictures(path, testingAmount = 0.8):
    megaList = []
    X = []
    Y = []
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if not file.__contains__('db'):
                img = Image.open(path)

                img = img.resize((196, 236), Image.ANTIALIAS)
                pic = np.array(img)
                pic = rgb2gray(pic)
                pic = pic / 255
                label = root.split("\\")
                #print(pic.shape)
                megaList.append((pic, label[len(label)-1]))
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


def loader(path, testingAmount = 0.8):
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            sound = audiosegment.from_file(path)
            #X.append(sound)
            hist_bins, times, amplitudes = sound.spectrogram(window_length_s = 0.03, overlap = 0.5)
            hist_bins_khz = hist_bins / 1000
            amplitudes_real_normed = np.abs(amplitudes) / len(amplitudes)
            amplitudes_logged = 10 * np.log10(amplitudes_real_normed + 1e-9)  # for numerical stability
            x, y = np.mgrid[:len(times), :len(hist_bins_khz)]
            fig, ax = plt.subplots()
            ax.pcolormesh(x, y, np.swapaxes(amplitudes_logged, 0, 1))
            label = root.split("\\")
            plt.savefig(file.split('.')[0]+"_"+ label[len(label)-1] + ".png")
            plt.close()
    return
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