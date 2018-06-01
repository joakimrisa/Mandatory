import os
from PIL import Image
import numpy as np
import autoencoder


def Luls(path, path2, testingAmount=0.8, n=10):
    row, col = 1440, 1920
    X = []
    X_salt = []
    for i in range(20):
        print("i: "+str(i))
        for root, dirs, files in os.walk(path):
            for file in files:
                path = os.path.join(root, file)
                if not file.__contains__('db'):
                    label = root.split("/")
                    with Image.open(path) as f:
                        x = np.array(f.convert("RGB").resize((row, col))) / 255
                    X.append(x)
                    number = file.strip('LPImage')
                    number = number.split('_')
                    length = str(number[3]).strip('.pn')
                    number = number[0] + '_' + number[1]
                    name = "Image" + number + "_CD_" + length + ".png"
                    pathJoined = os.path.join(path2, name)
                    with Image.open(pathJoined) as f:
                        x = np.array(f.convert("RGB").resize((row, col))) / 255
                    X_salt.append(x)

                    if X.__len__() == n:
                        x_Train = np.array(X[:int(X.__len__() * testingAmount)])
                        x_Test = np.array(X[x_Train.__len__():])
                        x_Train_Salt = np.array(X_salt[:int(X_salt.__len__() * testingAmount)])
                        x_Test_Salt = np.array(X_salt[x_Train.__len__():])
                        autoencoder.autoKongen(x_Train, x_Test,x_Train_Salt, x_Test_Salt)
                        X.clear()
                        X_salt.clear()


                        # Luls('training/pictures/LP_Pictures_1.0_300')

Luls('training/pictures/LP_Pictures_1.0_300', 'training/pictures/CD_Pictures_1.0_300')