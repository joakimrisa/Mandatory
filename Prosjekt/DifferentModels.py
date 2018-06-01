from keras.callbacks import TensorBoard
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import loader
import Models
import preprocessing
import os
import audiosegment
import pydub
from scipy.io import wavfile

'''
This file contains some of the different models used in the project.
'''



def DenseModel():

    x_train, x_train_noisy, x_test, x_test_noisy = loader.multipleLoad('training/music_splitted/CD_0.001', 'training/music_splitted/LP_0.001', limit=50000)

    Model = Models.denseModelFidjeSCIPYWAV()
    #arrayModel.load_weights('3000E_100000Samples_DenseFidje_Array.h5py')
    Model.compile(optimizer='adam', loss='binary_crossentropy')

    for i in range(1, 1001):
        baseLine = 100
        if os.path.exists('SCIPYWAV_'+str(100*i-1)+'E_100000Samples_Dense_Array.h5py'):
            Model.load_weights('SCIPYWAV_'+str(100*i-1)+'E_100000Samples_NYAGOA_DenseFidje_Array.h5py')
            Model.fit(x_train_noisy, x_train, batch_size=512, epochs=baseLine, verbose=2, validation_data=(x_test_noisy, x_test))
            Model.save('SCIPYWAV_'+str(100*i-1)+'E_100000Samples_NYAGOA_DenseFidje_Array.h5py')
    #arrayModel.load_weights('SCIPYWAV_' + str(100 * 64 - 1) + 'E_100000Samples_NYAGOA_DenseFidje_Array.h5py')
    #predikim = arrayModel.predict(tryToPred.reshape(1000, 192))

    #print(predikim*2147483647)
    #print(shouldBe*2147483647)

    #predikim = wavfile.write('mordi.wav', 96000, predikim)
    #shouldBe = wavfile.write('mordiskavær.wav', 96000, shouldBe)

DenseModel()
def autoEncoder():
    row, col = 1440, 1920 #280, 360

    (x_train, _), (x_test, _) = loader.loaderPictures('training/pictures/LP_Pictures_1.0_300', row=row, col=col, rgb=True, limit=(True, 10))#mnist.load_data()
    (x_train_noisy, _), (x_test_noisy, _) = loader.loaderPictures('training/pictures/CD_Pictures_1.0_300', row=row, col=col, rgb=True, limit=(True, 10))#mnist.load_data()

    x_train = preprocessing.reshape(x_train, row, col, True)
    x_test = preprocessing.reshape(x_test, row, col, True)

    x_train_noisy = preprocessing.reshape(x_train_noisy, row, col, True)
    x_test_noisy = preprocessing.reshape(x_test_noisy, row, col, True)

    autoencoder = Models.rgbModel(row, col)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train_noisy, x_train,
                    epochs=30,
                    batch_size=1,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

    autoencoder.save('1sec300_1440x1920_color.h5py')

#autoEncoder()
def autoKongen(xTrain, xTest, xTrain_salt, xTest_salt, row=1440, col=1920):
    x_train, x_test = xTrain, xTest
    x_train_noisy, x_test_noisy = xTrain_salt, xTest_salt

    x_train = preprocessing.reshape(x_train, row, col, True)
    x_test = preprocessing.reshape(x_test, row, col, True)

    x_train_noisy = preprocessing.reshape(x_train_noisy, row, col, True)
    x_test_noisy = preprocessing.reshape(x_test_noisy, row, col, True)

    autoencoder = Models.rgbModel(row, col)
    if os.path.exists('1sec300_1440x1920_color_20R.h5py'):
        autoencoder.load_weights('1sec300_1440x1920_color_20R.h5py')
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=1,
                    batch_size=1,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

    autoencoder.save('1sec300_1440x1920_color_20R.h5py')