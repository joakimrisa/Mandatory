import numpy as np
#import audiosegment
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import audiosegment
from scipy.io import wavfile
from bitstring import BitArray

def multipleLoad(path1, path2, trainingAmount=0.8, limit=1000):
    '''
    Loads multiple soundfiles. CD and his corresponding LP song.
    '''
    X = []
    Y = []
    for root, dirs, files in os.walk(path1):
        if X.__len__() == limit:
            break
        for file in files:
            if X.__len__() == limit:
                break
            path = os.path.join(root, file)
            #print(file)
            #fs, sound = wavfile.read(path)
            sound = load(path)
            #sound2 = audiosegment.from_file(path)

            Y.append(sound)
            #print("LP"+file)
            path = os.path.join(path2, "LP"+file)
            #fs, sound = wavfile.read(path)
            sound = load(path)
            X.append(sound)

    #number = 2147483647
    x_Train = np.array(X[:int(X.__len__() * trainingAmount)]) #/ number
    y_Train = np.array(Y[:int(Y.__len__() * trainingAmount)]) #/ number
    x_Test = np.array(X[x_Train.__len__():]) #/ number
    y_Test = np.array(Y[y_Train.__len__():]) #/ number

    return (x_Train, y_Train, x_Test, y_Test)

#multipleLoad('training/music_splitted/CD_0.001', 'training/music_splitted/LP_0.001')

def convertFromBytesToBin(bytes):
    bits = str(BitArray(bytes=bytes).bin)
    x = list(bits)
    x = np.array(x, dtype=int)
    return x

def convertFromBitsToInt(bits):
    bitString = ""
    for i in bits:
        bitString += str(i)
    bytes = BitArray(bin=bitString).bytes
    intValue = int.from_bytes(bytes, signed=True, byteorder='little')

    return intValue

def load(file):
    with open(file, 'rb') as f:
        Header = f.read(44)
        Data = f.read(768)
    music = wavfile.read(file)

    listOfData = []
    for i in range(0, 192):
        #listOfData.append(convertFromBitsToInt(convertFromBytesToBin(Data[4*i:4*i+4])))
        listOfData.append(convertFromBytesToBin(Data[4*i:4*i+4]))

    data = np.array(listOfData)
    data = data.reshape(data.shape[0]*data.shape[1], )
    return data


#load('training/music_splitted/CD_0.001/Image [1]_1.wav')

def loadArray(path, trainingAmount=0.8, limit=1000):
    '''
    Loads the value array from a song.
    '''
    X = []
    Y = []
    for root, dirs, files in os.walk(path):
        if X.__len__() == limit:
            break
        for file in files:
            if X.__len__() == limit:
                break
            path = os.path.join(root, file)
            fs, sound = wavfile.read(path)
            #sound = audiosegment.from_file(path)
            label = root.split("/")
            #print(sound.shape)
            X.append(np.reshape(sound, (192,)))
            #X.append(sound)
            Y.append(label[len(label)-1])
    x_Train = np.array(X[:int(X.__len__() * trainingAmount)])
    y_Train = np.array(Y[:int(Y.__len__() * trainingAmount)])
    x_Test = np.array(X[x_Train.__len__():])
    y_Test = np.array(Y[y_Train.__len__():])
    return (x_Train, y_Train), (x_Test, y_Test)


def loaderPictures(path, testingAmount = 0.8, row =28, col=28, shuffle=False, rgb=False, limit=(False, -1)):
    '''
    Loads pictures.
    '''
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
                    if not rgb:
                        megaList.append((((rgb2gray(np.array(f.resize((row, col)))))/255), label[len(label)-1]))
                    else:
                        megaList.append(((np.array(f.convert("RGB").resize((row, col))) / 255), label[len(label) - 1]))
                c+=1
                if limit[0] == True and limit[1] <= c:
                    break
        if limit[0] == True and limit[1] <= c:
            break
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