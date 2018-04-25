import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.layers import RNN as RNNL
from keras.layers import SimpleRNNCell
from keras.layers.embeddings import Embedding
import os
import loader

#charX, y, numberOfCharsToLearn, numberOfUniqueChars, idsForChars = loader.forFolder("data/comm_use.0-9A-B.txt", 1)




#number = 40000
def charRNN(charX, y, numberOfCharsTolearn, numberOfUniqueChars, idsForChars, name, epochs=40, batch_size = 512, train = True, verification = True ):

    X = np_utils.to_categorical(charX, num_classes=numberOfUniqueChars)
    y = np_utils.to_categorical(y, num_classes=numberOfUniqueChars)


    model = Sequential()
    cell = SimpleRNNCell(32)
    #x = keras.Input((None, 5))
    #layer = RNNL(cell)
    #y = layer(x)
    model.add(RNNL(cell, input_shape=(X.shape[1], X.shape[2])))
    #model.add(Dropout(0.20))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    if os.path.exists(name):
        model.load_weights(name)
    if train:
        model.fit(X, y, epochs=epochs, batch_size=batch_size)
        model.save_weights(name)
    if verification:

        randomVal = np.random.randint(0, len(charX)-1)
        randomStart = list(charX[randomVal])
        print("".join([idsForChars[value] for value in randomStart]))
        for i in range(100):
            X = np_utils.to_categorical(randomStart, num_classes=numberOfUniqueChars)
            X = np.expand_dims(X, axis=0)
            pred = model.predict(X)
            index = np.argmax(pred)
            randomStart.append(index)
            randomStart = randomStart[1: len(randomStart)]

        print(randomStart)
        print("".join([idsForChars[value] for value in randomStart]))


def charLSTM(charX, y, numberOfCharsTolearn, numberOfUniqueChars, idsForChars, name, epochs=40, batch_size = 512, train = True, verification = True ):

    X = np_utils.to_categorical(charX, num_classes=numberOfUniqueChars)
    y = np_utils.to_categorical(y, num_classes=numberOfUniqueChars)


    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    #model.add(Dropout(0.20))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    if os.path.exists(name):
        model.load_weights(name)
    if train:
        model.fit(X, y, epochs=epochs, batch_size=batch_size)
        model.save_weights(name)
    if verification:

        randomVal = np.random.randint(0, len(charX)-1)
        randomStart = list(charX[randomVal])
        print("".join([idsForChars[value] for value in randomStart]))
        for i in range(100):
            X = np_utils.to_categorical(randomStart, num_classes=numberOfUniqueChars)
            X = np.expand_dims(X, axis=0)
            pred = model.predict(X)
            index = np.argmax(pred)
            randomStart.append(index)
            randomStart = randomStart[1: len(randomStart)]

        print(randomStart)
        print("".join([idsForChars[value] for value in randomStart]))

def wordLSTM(charX, y, numberOfUniqueChars, idsForChars, name, epochs=40, batch_size = 512, train = True, verification = True ):

    #X = np_utils.to_categorical(charX, num_classes=numberOfUniqueChars)
    #X = np.array(charX)
    X = charX
    print(X.shape)
    y = np_utils.to_categorical(y, num_classes=numberOfUniqueChars)
    #print(X.shape)
    max_features = numberOfUniqueChars

    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(256))
    #model.add(Dropout(0.20))
    model.add(Dense(numberOfUniqueChars, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists(name):
        model.load_weights(name)
    if train:
        model.fit(X, y, epochs=epochs, batch_size=batch_size)
        model.save_weights(name)
    if verification:

        randomVal = np.random.randint(0, len(charX)-1)
        randomStart = list(charX[randomVal])
        print(randomStart)
        print(" ".join([idsForChars[value] for value in randomStart]))
        for i in range(5):
            #X = np_utils.to_categorical(randomStart, num_classes=numberOfUniqueChars)
            #X = np.expand_dims(X, axis=0)
            X = np.array(randomStart)
            pred = model.predict(X)
            index = np.argmax(pred)
            randomStart.append(index)
            randomStart = randomStart[1: len(randomStart)]

        print(randomStart)
        print(" ".join([idsForChars[value] for value in randomStart]))


#number = 40000
def wordRNN(charX, y, numberOfUniqueChars, idsForChars, name, epochs=40, batch_size = 512, train = True, verification = True ):

    X = np_utils.to_categorical(charX, num_classes=numberOfUniqueChars)
    y = np_utils.to_categorical(y, num_classes=numberOfUniqueChars)

    max_features = 20000
    model = Sequential()
    cell = SimpleRNNCell(32)
    #x = keras.Input((None, 5))
    #layer = RNNL(cell)
    #y = layer(x)
    model.add(Embedding(max_features, 128))
    model.add(RNNL(cell, input_shape=(X.shape[1], X.shape[2])))
    #model.add(Dropout(0.20))
    model.add(Dense(y.shape[1], activation='softmax'))   # Må kanskje bort
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    if os.path.exists(name):
        model.load_weights(name)
    if train:
        model.fit(X, y, epochs=epochs, batch_size=batch_size)
        model.save_weights(name)
    if verification:

        randomVal = np.random.randint(0, len(charX)-1)
        randomStart = list(charX[randomVal])
        print("".join([idsForChars[value] for value in randomStart]))
        for i in range(100):
            X = np_utils.to_categorical(randomStart, num_classes=numberOfUniqueChars)
            X = np.expand_dims(X, axis=0)
            pred = model.predict(X)
            index = np.argmax(pred)
            randomStart.append(index)
            randomStart = randomStart[1: len(randomStart)]

        print(randomStart)
        print("".join([idsForChars[value] for value in randomStart]))
