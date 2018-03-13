import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import os
import loader

#charX, y, numberOfCharsToLearn, numberOfUniqueChars, idsForChars = loader.forFolder("data/comm_use.0-9A-B.txt", 1)




#number = 40000
def RNN(charX, y, numberOfCharsTolearn, numberOfUniqueChars, idsForChars, name, epochs=40, batch_size = 512, train = True ):

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

    '''
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
    '''