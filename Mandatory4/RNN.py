import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import loader

charX, y, numberOfCharsToLearn, numberOfUniqueChars, idsForChars = loader.forFolder("data/comm_use.0-9A-B.txt", 1)




number = 70000
charX = np.array(charX[:number])

y = y[:number]

print(charX)


X = np_utils.to_categorical(charX, num_classes=numberOfUniqueChars)

#x = np.reshape(charX, (len(charX), numberOfCharsToLearn, 1))
#print(x.shape)
print(X)

#X = X/float(numberOfUniqueChars)
#print(y)

y = np_utils.to_categorical(y)
#print(y)
print(X.shape[0])
print(X.shape[1])

model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
#model.add(Dropout(0.20))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.load_weights("Othello.hdf6")
model.fit(X, y, epochs=30, batch_size=1024)
model.save_weights("Othello.hdf6")


randomVal = np.random.randint(0, len(charX)-1)
randomStart = list(charX[randomVal])
print(randomVal)

print("".join([idsForChars[value] for value in randomStart]))
for i in range(100):
    #x = np.reshape(randomStart, (1, len(randomStart), 1))
    #print(randomStart)
    X = np_utils.to_categorical(randomStart, num_classes=numberOfUniqueChars)
    #print(X.shape)
    X = np.expand_dims(X, axis=0)
    #print(X.shape)
    #x = x/float(numberOfUniqueChars)
    pred = model.predict(X)
    index = np.argmax(pred)
    randomStart.append(index)
    randomStart = randomStart[1: len(randomStart)]

print(randomStart)
print("".join([idsForChars[value] for value in randomStart]))