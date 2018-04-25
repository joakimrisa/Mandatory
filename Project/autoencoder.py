from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dense, UpSampling2D, Input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import loader

row, col = 200, 240
(x_train, _), (x_test, _) = loader.loaderPictures('training/pictures/LP_Pictures_1.0_300', row=row, col=col)#mnist.load_data()
(x_train_noisy, _), (x_test_noisy, _) = loader.loaderPictures('training/pictures/CD_Pictures_1.0_300', row=row, col=col)#mnist.load_data()

print(x_train.shape)


x_train = np.reshape(x_train, (len(x_train), row, col, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), row, col, 3))  # adapt this if using `channels_first` image data format

x_train_noisy = np.reshape(x_train_noisy, (len(x_train_noisy), row, col, 3))  # adapt this if using `channels_first` image data format
x_test_noisy = np.reshape(x_test_noisy, (len(x_test_noisy), row, col, 3))  # adapt this if using `channels_first` image dat

'''
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
'''
n = 10
plt.figure(figsize=(20, 2))

'''
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
'''

input_img = Input(shape=(row, col, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', shape=input_img.shape)(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train,
                epochs=30,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

autoencoder.save('1sec300_300x340_color.h5py')

