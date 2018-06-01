from __future__ import print_function
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
import numpy as np
import os
import audiosegment

'''
This is a failed attempt of LSTM. 
'''

batch_size = 128  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

CD_PATH = 'training\music_splitted\Test\CD_0.001\Training'
LP_PATH = 'training\music_splitted\Test\LP_0.001\Training'

CD = []
LP = []
C = 100001
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



encoder_inputs = Input(shape=(None, 1))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)


encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None, 1))


decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

decoder_dense = Dense(1, activation='relu')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='mean_squared_error')


model.load_weights('s2s100000.h5')
