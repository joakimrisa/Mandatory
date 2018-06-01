from __future__ import print_function
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
import numpy as np
import os
import audiosegment


batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

CD_PATH = 'training\music_splitted\Test\CD_0.001\Training'
LP_PATH = 'training\music_splitted\Test\LP_0.001\Training'

CD = []
LP = []
C = 10000
for root, dirs, files in os.walk(CD_PATH):
    for file in files:
        path = os.path.join(root, file)
        sound = audiosegment.from_file(path)
        if sound.spl.__len__() == 88:
            newList = list(sound.spl)
            newList.append(0.0)
            newList.append(0.0)
            CD.append(np.array(newList))
        else:
            CD.append(np.array(sound.spl))
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
#CD = CD / CD.max()
#CD = np.log(CD)
#CD = preprocessing.normalize(CD)
CD = CD.reshape(CD.shape[0], CD.shape[1], 1)
LP = np.array(LP)
#LP = LP / LP.max()
#LP = np.log(LP)
#LP = preprocessing.normalize(LP)
#LP = LP.reshape(LP.shape[0],LP.shape[1], 1)

model = Sequential()
model.add(LSTM(80, return_sequences=True, input_shape=(90, 1)))
model.add(LSTM(160))
model.add(Dense(192, activation='relu'))
print(model.summary())
# compile network
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# fit network
if os.path.exists('LSTM_2LSTM_10000Samples_500E.h5'):
    model.load_weights('LSTM_2LSTM_10000Samples_500E.h5')
model.fit(CD, LP, batch_size=batch_size, epochs=500, verbose=2)
model.save('LSTM_2LSTM_10000Samples_500E.h5')
#model.load_weights('RNN.h5')
#pred = model.predict(trainingJoa)
#print(pred)
#print(faktiskJoa)