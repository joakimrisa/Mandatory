from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dense, UpSampling2D, Input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import loader
from PIL import Image
import Models
import preprocessing

'''
This is used to test the models after training. 
'''

def to_rgb2(im):
    # as 1, but we use broadcasting in one line
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret


row, col = 1440, 1920

(x_train_noisy, _), (x_test_noisy, _) = loader.loaderPictures('training/pictures/CD_test', row=row, col=col, rgb=True)#mnist.load_data()

x_train_noisy = preprocessing.reshape(x_train_noisy, row, col, True)
x_test_noisy = preprocessing.reshape(x_test_noisy, row, col, True)

autoencoder = Models.rgbModel(row, col)
autoencoder.load_weights('1sec300_1440x1920_color_20R.h5py')

x_decoded = autoencoder.predict(x_test_noisy)

imgs = x_decoded[:1]
imgs = imgs.reshape((row, col, 3))
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.imshow(imgs, interpolation='nearest')
Image.fromarray(imgs).save('1sec300_1920x1440_.png')
plt.show()
