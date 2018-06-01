from __future__ import print_function
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
import numpy as np
import os
import audiosegment

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

trainingJoa = np.array(CD[100000:]).reshape(1, 90, 1)
faktiskJoa = np.array(LP[100000:]).reshape(1, 192, 1)
CD = np.array(CD)
CD = CD.reshape(CD.shape[0], CD.shape[1], 1)
LP = np.array(LP)
LP = LP.reshape(LP.shape[0],LP.shape[1], 1)

print(CD.shape)
print(LP.shape)
'''for i in range(CD.__len__()):
    CDTOLP.append([CD[i], LP[i]])
    LPTOCD.append([LP[i], CD[i]])
'''

test = "hei"

#model = Sequential()
#model.add(LSTM(latent_dim, return_state=True, input_shape=(1000,90)))
#model.add(LSTM(latent_dim, return_sequences=True, return_state=True))
#model.add(Dense(120, activation='softmax'))

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, 1))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, 1))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return state  s in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

decoder_dense = Dense(1, activation='relu')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='mean_squared_error')

#model.fit([CD, LP], LP,batch_size=batch_size,epochs=epochs, validation_split=0.2)
# Save model
#model.save('s2s100000.h5')

model.load_weights('s2s100000.h5')
testLOL = np.zeros((1, 192, 1))
heimann = model.predict([trainingJoa, testLOL])
print(heimann)
print(faktiskJoa)