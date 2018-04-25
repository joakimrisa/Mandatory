from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import random
import keras
import matplotlib.pyplot as plt
import lfwLoader
import os

def modellen():
        model = Sequential()
        model.add(Conv2D(32,
                         kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64,
                         kernel_size=(3, 3),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='sigmoid'))
        model.compile(
                loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

        return model

batch_size = 128
num_classes = 2
epochs = 12



img_rows, img_cols = 64,64
(x_train,y_train),(x_test,y_test) = lfwLoader.loader(folder="nyeTest1", trainingAmount=0)#mnist.load_data()
#print(x_train)

#print(type(x_train))

#print(x_train[200])

'''
for x in range(100):
        n = random.randint(0,len(x_train))
        #plt.plot(10,10,x)
        plt.axis('off')
        plt.imshow(x_train[n].reshape(28,28),cmap='gray')
plt.show()
'''
x_train = x_train
print(x_train)
print(x_train.shape)
y_train = y_train

x_train = x_train.reshape(x_train.shape[0],img_rows, img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
input_shape = (img_rows,img_cols,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:',x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

model = modellen()
if os.path.exists("weights.h5py"):
        model.load_weights("weights.h5py")
#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print(score)
#model.save("weights.h5py")
