from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import random
import keras
import matplotlib.pyplot as plt
import loader

img_rows = 60
img_cols = 80

(x_train,y_train),(x_test,y_test) = loader.loaderPictures("training\pictures")

x_train = x_train
print(x_train)
print(x_train.shape)
y_train = y_train

x_train = x_train.reshape(x_train.shape[0],img_rows, img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols,1)
input_shape = (img_rows,img_cols,1)


print('x_train shape:',x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

model = Sequential()
model.add(Conv2D(32,
        kernel_size=(3,3),
        activation='relu',
        input_shape=input_shape))
model.add(Conv2D(64,
        kernel_size=(3,3),
        activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=30,verbose=1,validation_data=(x_test,y_test))
score = model.evaluate(x_test,y_test,verbose=0)

