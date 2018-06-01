
import re
import os
import rnnFunction




def forFolder(path, n=10, maxValue= 45000):
    '''
    This function splits the article to be forwarded to the char networks
    '''
    data = []
    c = 0
    chars = set()
    numberOfUniqueChars = 0
    for root, dirs, files in os.walk(path):
        if c == n:
            break
        for file in files:
            if c == n:
                break
            with open(os.path.join(root, file), 'r', errors='ignore') as f:
                data+= f.read().lower()
            chars=list(set().union(sorted(list(set(data)))))
            c+=1
    numberOfUniqueChars = len(chars)
    data = []
    c = 0
    print(numberOfUniqueChars)
    for i in range(0, 20):
        for root, dirs, files in os.walk(path):
            for file in files:
                if c==n:
                    return
                with open(os.path.join(root, file), 'r', errors='ignore') as f:
                    data += f.read().lower()
                if data.__len__() >= maxValue:
                    chars = sorted(list(set(data[:maxValue])))
                    totalChars = len(data[:maxValue])

                    CharsForids = {char: Id for Id, char in enumerate(chars)}
                    idsForChars = {Id: char for Id, char in enumerate(chars)}
                    numberOfCharsToLearn = 100

                    counter = totalChars - numberOfCharsToLearn
                    charX = []
                    y = []
                    for i in range(0, counter, 1):
                        theInputChars = data[i:i + numberOfCharsToLearn]
                        theOutputChars = data[i + numberOfCharsToLearn]
                        charX.append(([CharsForids[char] for char in theInputChars]))
                        y.append(CharsForids[theOutputChars])
                    rnnFunction.charLSTM(name="1EpochBulk_5F_20R.hdf5", charX=charX, y=y, numberOfCharsTolearn=numberOfCharsToLearn, numberOfUniqueChars=numberOfUniqueChars, idsForChars=idsForChars, train=True, epochs=1)
                    rnnFunction.charRNN(name="1EpochBulk_7F_20R.hdf5", charX=charX, y=y,
                                         numberOfCharsTolearn=numberOfCharsToLearn,
                                         numberOfUniqueChars=numberOfUniqueChars, idsForChars=idsForChars, train=True,
                                         epochs=1)
                    data = data[maxValue:]
                #c += 1
forFolder('comm_use.C-H.txt', n=3)