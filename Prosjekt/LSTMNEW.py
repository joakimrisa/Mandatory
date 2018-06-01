from __future__ import print_function
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
import numpy as np
import os
from scipy.io import wavfile
import audiosegment

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

CD_PATH = 'training\Alligned Chunks\CD'
LP_PATH = 'training\music_splitted\Test\LP_0.001\Training'

CD = []
LP = []
C = 10000
for root, dirs, files in os.walk(CD_PATH):
    for file in files:
        path = os.path.join(root, file)
        sound = wavfile.read(path)
        CD.append(sound[1])
        if CD.__len__() == C:
            break
    if CD.__len__() == C:
        break

for root, dirs, files in os.walk(LP_PATH):
    for file in files:
        path = os.path.join(root, file)
        sound = wavfile.read(path)
        LP.append(np.array(sound[1]))
        if LP.__len__() == C:
            break
    if LP.__len__() == C:
        break


model = Sequential()
model.add(LSTM(80, return_sequences=True, input_shape=(192, 1)))
model.add(LSTM(160, return_sequences=True))
model.add(LSTM(80))
model.add(Dense(1, activation='relu'))

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