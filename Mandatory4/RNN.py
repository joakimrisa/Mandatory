import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import loader

charX, y, numberOfCharsToLearn, numberOfUniqueChars, idsForChars = loader.forFolderWords("comm_use.C-H.txt", 40)
print("wallah")
X = np.reshape(charX, (len(charX), numberOfCharsToLearn, 1))

X = X/float(numberOfUniqueChars)


y = np_utils.to_categorical(y)
print(y)


model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.20))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='cate,'
                   'gorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=5, batch_size=256)
model.save_weights("Othello.hdf5")
#model.load_weights("Othello.hdf5")

randomVal = np.random.randint(0, len(charX)-1)
randomStart = charX[randomVal]

ompa = 0
for i in range(4):
    x = np.reshape(randomStart, (1, len(randomStart), 1))
    x = x/float(numberOfUniqueChars)
    #print(x)
    pred = model.predict(x)
    index = np.argmax(pred)
    randomStart.append(index)
    randomStart = randomStart[1: len(randomStart)]
    ompa = randomStart

print(ompa)
print(index)
print(" ".join([idsForChars[value] for value in randomStart]))