import numpy as np
import audiosegment
import os
import random
import matplotlib.pyplot as plt
from PIL import Image


def dice(song, millisecond):
    length = int(len(song)/(millisecond))
    songs = []
    for i in range(length):
        songs.append(song[millisecond*i:millisecond*i+millisecond])
    return songs

def split():
    for root, dirs, files in os.walk("training/music_org/CD"):
        for file in files:
            path = os.path.join(root, file)
            print(path)
            print(root)
            sound = audiosegment.from_file(path)
            # len() and slicing are in milliseconds
            sounds = dice(sound, 1000)
            i=1
            for song in sounds:
                filename = file.split(".")
                folder = os.path.basename("training")
                folder2 = 'music_splitted'
                folder3 = 'CD_1.0'
                path = os.path.join(folder,folder2)
                path = os.path.join(path, folder3)
                path = os.path.join(path, str(filename[0] + "_" + str(i) + ".wav"))
                print(path)
                song.export(path, format="wav")
                i += 1
#split()


def createPictures(path, savePath):
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            sound = audiosegment.from_file(path)
            hist_bins, times, amplitudes = sound.spectrogram(window_length_s = 0.03)
            hist_bins_khz = hist_bins / 1000
            amplitudes_real_normed = np.abs(amplitudes) / len(amplitudes)
            amplitudes_logged = 10 * np.log10(amplitudes_real_normed + 1e-9)  # for numerical stability
            x, y = np.mgrid[:len(times), :len(hist_bins_khz)]
            fig, ax = plt.subplots()
            ax.set_position([0,0,1,1])
            ax.pcolormesh(x, y, np.swapaxes(amplitudes_logged, 0, 1))
            plt.axis('off')
            label = root.split("/")
            plt.savefig(os.path.join(savePath, file.split('.')[0] + "_" + label[len(label) - 1] + ".png"), dpi = 300)
            plt.close()
    return

createPictures('training/music_splitted/LP_1.0', 'training/pictures/LP_pictures_1.0_300')


def fromStringToInt(string):
    if string == "CD_quality":
        return 0
    else:
        return 1

def fromIntToString(int):
    if int == 0:
        return "CD_quality"
    else:
        return "LP_quality"

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray