from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dense, UpSampling2D, Input
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Reshape

def denseModelFidjeSCIPYWAV():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(6144, )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(512, activation='relu'))
    #model.add(Reshape((192, )))
    model.add(Dense(6144, activation='relu'))
    return model


def denseModelFidje2():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(192,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(192, activation='relu'))

    return model

def denseModelFidje():

    model = Sequential()
    model.add(Dense(32, activation='relu',input_shape=(192,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(192, activation='relu'))

    return model

def denseModel():

    input_array = Input(shape=(192,))

    encoded = Dense(32, activation='relu')(input_array)

    decoded = Dense(192, activation='relu')(encoded)

    autoencoder = Model(input_array, decoded)

    encoded_input = Input(shape=(32,))

    decoder_layer = autoencoder.layers[-1]

    decoder = Model(encoded_input, decoder_layer(encoded_input))

    return (autoencoder, decoder)


def rgbModel(row, col):

    input_img = Input(shape=(row, col, 3))  # adapt this if using `channels_first` image data format

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    return autoencoder

def grayScale(row, col):
    input_img = Input(shape=(row, col, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    return autoencoder
