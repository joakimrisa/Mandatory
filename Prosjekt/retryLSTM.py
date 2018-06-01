from __future__ import print_function
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
import numpy as np
import os
import audiosegment
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

CD_PATH = 'training\music_splitted\Test\CD_0.001\Training'
LP_PATH = 'training\music_splitted\Test\LP_0.001\Training'

CD = []
LP = []
C = 1000
for root, dirs, files in os.walk(CD_PATH):
    for file in files:
        path = os.path.join(root, file)
        sound = audiosegment.from_file(path)
        if sound.spl.__len__() == 88:
            newList = list(sound.spl)
            newList.append(0.0)
            newList.append(0.0)
            CD.append(np.array(newList+ [0 for _ in range(102)]))
        else:
            newList = list(sound.spl)
            CD.append(np.array(newList + [0 for _ in range(102)]))
        if CD.__len__() == C:
            break
    if CD.__len__() == C:
        break

for root, dirs, files in os.walk(LP_PATH):
    for file in files:
        path = os.path.join(root, file)
        sound = audiosegment.from_file(path)
        LP.append(np.array(sound.spl))
        if LP.__len__() == C:
            break
    if LP.__len__() == C:
        break

#trainingJoa = np.array(CD[1000:]).reshape(1, 90, 1)
#faktiskJoa = np.array(LP[1000:]).reshape(1, 192)
CD = np.array(CD)
#CD = np.log(CD)
#CD = preprocessing.normalize(CD)


CD = CD.reshape(CD.shape[0], CD.shape[1], 1)
print(CD.shape)
LP = np.array(LP)
LP = LP.reshape(LP.shape[0], LP.shape[1], 1)
print(LP.shape)

model = Sequential()
model.add(LSTM(150, input_shape=(192, 1)))
model.add(RepeatVector(192))
model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='relu')))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
model.fit(CD, LP, epochs=100, verbose=2)
