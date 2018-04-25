
import re
import os
import rnnFunction
from keras.preprocessing.text import Tokenizer
import numpy as np




def forFolder(path, n=10, maxValue= 45000):
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
            print(data)
            #words = set().union(words,set(data))
            c += 1

    data = " ".join(data)
    words = sorted(words)
    print(words)
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

    numberOfUniqueWords = len(words)

    rnnFunction.wordLSTM(name="1EpochBulk_15F_20R.hdf5", charX=X, y=y,
                         numberOfUniqueChars=vocabsize, idsForChars=idsForWords, train=True,
                         epochs=60, verification=False)
    '''
    data = []
    c = 0
    print(numberOfUniqueWords)
    for i in range(0, 20):
        c = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                data = []
                if c == n:
                    break
                with open(os.path.join(root, file), 'r', errors='ignore') as f:
                    for lines in f:
                        for word in lines.split():
                            data.append(word.lower())
                    #data += f.read().lower()
                #if data.__len__() >= maxValue:
                Words = sorted(list(set(data)))
                totalWords = len(data)


                numberOfWordsToLearn = 5

                counter = data.__len__() - numberOfWordsToLearn
                WordsX = []
                y = []
                for i in range(0, counter, 1):
                    theInputWords = data[i:i+numberOfWordsToLearn]
                    theOutputWords = data[i+numberOfWordsToLearn]
                    WordsX.append(([WordsForids[word] for word in theInputWords]))
                    y.append(WordsForids[theOutputWords])
                #rnnFunction.charLSTM(name="1EpochBulk_5F_20R.hdf5", charX=charX, y=y, numberOfCharsTolearn=numberOfCharsToLearn, numberOfUniqueChars=numberOfUniqueChars, idsForChars=idsForChars, train=True, epochs=1)
                rnnFunction.wordLSTM(name="1EpochBulk_15F_20R.hdf5", charX=WordsX, y=y,
                                         numberOfUniqueChars=numberOfUniqueWords, idsForChars=idsForWords, train=True,
                                         epochs=1)
                c += 1
            if c == n:
                break
                '''
forFolder('comm_use.C-H.txt', n=15)