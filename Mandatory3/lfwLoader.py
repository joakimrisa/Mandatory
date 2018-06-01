import os
from PIL import Image
import numpy as np
import translate
from keras.preprocessing.image import ImageDataGenerator


def loader(trainingAmount = 0.8, folder = "gender"):
    '''
    This function loads in the data and splits it into training and testing and returns this.
    '''
    X = []
    Y = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            split = file.split('.')
            if not (split[1].__contains__('db')):
                im = Image.open(os.path.join(root,file))
                im = im.resize((64, 64), Image.ANTIALIAS)
                pic = np.array(im)
                pic = rgb2gray(pic)
                pic = pic/255
                X.append(pic)
                name = file.split('_')
                Y.append(translate.toNumber(name[2]))

    x_Train = np.array(X[:int(X.__len__()*trainingAmount)])
    y_Train = np.array(Y[:int(Y.__len__()*trainingAmount)])
    x_Train_Extended, y_Train_Extended = createImages(x_Train,y_Train)
    x_Train.append(x_Train_Extended)
    y_Train.append(y_Train_Extended)
    x_Test = np.array(X[x_Train.__len__():])
    y_Test = np.array(Y[y_Train.__len__():])

    return(x_Train,y_Train),(x_Test,y_Test)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def createImages(x_train, y_train):
    '''

    This function takes in X_train and y_train and generates more training data for the network.
    It then returns
    '''
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    x_train = x_train.astype("float32")
    imageListX = []
    imageListY = []

    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    datagen.fit(x_train)
    for xbatch,ybatch in datagen.flow(x_train, y_train, batch_size=x_train.__len__()):
        for i in range(xbatch.__len__()):
            im = Image.fromarray(xbatch[i].astype('uint8'))
            filename = str(str(ybatch[i][0]) + "_" + str(ybatch[i][1]) + "_" + str(ybatch[i][2]) + "_" + "FS" + str(ybatch[i][3]))
            if ybatch[i][2] == "male":

                im.save(os.path.join("newyeah/male", filename),'jpeg')
            else:
                im.save(os.path.join("newyeah/female", filename),'jpeg')
        break

    datagen = ImageDataGenerator(zca_whitening=True)
    datagen.fit(x_train)
    for xbatch, ybatch in datagen.flow(x_train, y_train, batch_size=x_train.__len__()):
        for i in range(xbatch.__len__()):
            imageListX.append(xbatch[i])
            imageListY.append(ybatch[i])
        break
    datagen = ImageDataGenerator(rotation_range=90)
    datagen.fit(x_train)
    for xbatch,ybatch in datagen.flow(x_train, y_train, batch_size=x_train.__len__()):
        for i in range(xbatch.__len__()):
            imageListX.append(xbatch[i])
            imageListY.append(ybatch[i])
        break
    datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
    datagen.fit(x_train)
    for xbatch,ybatch in datagen.flow(x_train, y_train, batch_size=x_train.__len__()):
        for i in range(xbatch.__len__()):
            imageListX.append(xbatch[i])
            imageListY.append(ybatch[i])
        break
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    datagen.fit(x_train)
    for xbatch,ybatch in datagen.flow(x_train, y_train, batch_size=x_train.__len__()):
        for i in range(xbatch.__len__()):
            imageListX.append(xbatch[i])
            imageListY.append(ybatch[i])
        break
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    datagen.fit(x_train)
    for xbatch,ybatch in datagen.flow(x_train, y_train, batch_size=x_train.__len__()):
        for i in range(xbatch.__len__()):
            imageListX.append(xbatch[i])
            imageListY.append(ybatch[i])
        break
    return (np.array(imageListX), np.array(imageListY))
