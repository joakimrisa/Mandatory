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
C = 1000
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

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, 1))
encoder = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, 1))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(encoder_outputs, initial_state=encoder_states)

decoder_dense = Dense(1, activation='relu')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(encoder_inputs, decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit(CD, LP, batch_size=batch_size, epochs=epochs, validation_split=0.2)
# Save model
model.save('s2s.h5')
'''
# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
'''