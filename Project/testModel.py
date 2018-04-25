from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dense, UpSampling2D, Input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import loader
from PIL import Image

def to_rgb2(im):
    # as 1, but we use broadcasting in one line
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret


row, col = 280, 320
#(x_train, _), (x_test, _) = loader.loaderPictures('training/pictures/LP_Pictures_1.0_600', row=row, col=col)#mnist.load_data()
(x_train_noisy, _), (x_test_noisy, _) = loader.loaderPictures('training/pictures/CD_test', row=row, col=col)#mnist.load_data()


#x_train = np.reshape(x_train, (len(x_train), row, col, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), row, col, 1))  # adapt this if using `channels_first` image data format

x_train_noisy = np.reshape(x_train_noisy, (len(x_train_noisy), row, col, 1))  # adapt this if using `channels_first` image data format
x_test_noisy = np.reshape(x_test_noisy, (len(x_test_noisy), row, col, 1))  # adapt this if using `channels_first` image dat


input_img = Input(shape=(row, col, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.load_weights('1sec300_300x340.h5py')

x_decoded = autoencoder.predict(x_test_noisy)

# Display the 1st 8 corrupted and denoised images
rows, cols = 10, 30
num = rows * cols
imgs = x_decoded[:1]
print(imgs.shape)
imgs = imgs.reshape((1, 1, row, col))
print(imgs.shape)
imgs = imgs.reshape((1, 1, row, col))
print(imgs.shape)
imgs = np.vstack([np.hstack(i) for i in imgs])
print(imgs.shape)
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
imgs = to_rgb2(imgs)
print(imgs.shape)
Image.fromarray(imgs).save('1sec300_300x340_.png')
plt.show()
