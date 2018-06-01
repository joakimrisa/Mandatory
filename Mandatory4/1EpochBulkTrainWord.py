
import re
import os
import rnnFunction
from keras.preprocessing.text import Tokenizer
import numpy as np




def forFolder(path, n=10, maxValue= 45000):
    '''
    This function splits the article to be forwarded to the word networks
    '''
    data = []
    c = 0
    words = set()
    numberOfUniqueWords = 0
    for root, dirs, files in os.walk(path):
        if c == n:
            break
        for file in files:
            if c == n:
                break
            with open(os.path.join(root, file), 'r', errors='ignore') as f:
                for lines in f:
                    for word in lines.split():
                        data.append(word.lower())
            c += 1

    data = " ".join(data)
    words = sorted(words)
    WordsForids = {word: Id for Id, word in enumerate(words)}
    idsForWords = {Id: word for Id, word in enumerate(words)}
    tokenize = Tokenizer()
    tokenize.fit_on_texts([data])
    encoded = tokenize.texts_to_sequences([data])[0]
    vocabsize = len(tokenize.word_index) + 1
    sequences = list()
    for i in range(1, len(encoded)):
        sequence = encoded[i - 1:i + 1]
        sequences.append(sequence)

    sequences = np.array(sequences)
    X, y = sequences[:, 0], sequences[:, 1]
    rnnFunction.wordRNN(name="RNN1EpochBulk_300F_20R.hdf5", charX=X, y=y,
                         numberOfUniqueChars=vocabsize, idsForChars=idsForWords, train=True,
                         epochs=60, verification=False)
    rnnFunction.wordLSTM(name="RNN1EpochBulk_300F_20R.hdf5", charX=X, y=y,
                         numberOfUniqueChars=vocabsize, idsForChars=idsForWords, train=True,
                         epochs=60, verification=False)

forFolder('comm_use.C-H.txt', n=15)