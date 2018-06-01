import numpy as np
import audiosegment
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import librosa
from librosa import display
import tqdm
import scipy
import soundfile
from scipy.io import wavfile
import time

def reshape(x, row, col, rgb=False):
    if rgb:
        x_train = np.reshape(x, (len(x), row, col, 3))
    else:
        x_train = np.reshape(x, (len(x), row, col, 1))

    return x_train


def dice(song, millisecond):
    '''
    Used to split the song.
    '''
    length = int(len(song) / (millisecond))
    songs = []
    for i in range(length):
        songs.append(song[millisecond * i:millisecond * i + millisecond])

    return songs


def splitAndOverlap():
    '''
    Splits and creates overlapping chunks.
    '''

    for root, dirs, files in os.walk("training/music_org/LPNEW(32)"):
        for file in files:
            path = os.path.join(root, file)
            print(path)
            print(root)
            sound = audiosegment.from_file(path)
            # len() and slicing are in milliseconds
            sounds = dice(sound, 2000)
            overlapKing = []
            k = 1
            for i in range(sounds.__len__()-1):
                if i == 0:
                    sound = sounds[i]+sounds[i+1][:sounds[i+1].__len__()/2]
                elif i < sounds.__len__()-2:
                    sound = sounds[i-1][sounds[i-1].__len__()/2:]+sounds[i]+sounds[i+1][:sounds[i+1].__len__()/2]
                else:
                    sound = sounds[i-1][sounds[i-1].__len__()/2:]+sounds[i]
                overlapKing.append(sound)

            for song in overlapKing:
                filename = file.split(".")
                folder = os.path.basename("training")
                folder2 = 'Ali'
                # folder3 = 'Test'
                folder4 = 'LP_4.0'
                # folder5 = 'Training'
                path = os.path.join(folder, folder2)
                # path = os.path.join(path, folder3)
                path = os.path.join(path, folder4)
                # path = os.path.join(path, folder5)
                path = os.path.join(path, str(filename[0] + "_" + str(k) + ".wav"))
                print(path)
                song.export(path, format="wav")
                k += 1

#splitAndOverlap()

def split():
    '''

    Split the song into chunks.
    '''

    for root, dirs, files in os.walk("training/music_org/PL"):
        for file in files:
            path = os.path.join(root, file)
            print(path)
            print(root)
            sound = audiosegment.from_file(path)
            # len() and slicing are in milliseconds
            sounds = dice(sound, 2000)
            i = 1
            for song in sounds:
                filename = file.split(".")
                folder = os.path.basename("training")
                folder2 = 'music_splitted'
                # folder3 = 'Test'
                folder4 = 'LP_0.001'
                # folder5 = 'Training'
                path = os.path.join(folder, folder2)
                # path = os.path.join(path, folder3)
                path = os.path.join(path, folder4)
                # path = os.path.join(path, folder5)
                path = os.path.join(path, str(filename[0] + "_" + str(i) + ".wav"))
                print(path)
                song.export(path, format="wav")
                i += 1


# librosaPictureCreator('training/music_splitted/CD_1.0', 'training/pictures/CD_Pictures_1.0_300')
def findPeeks(array, n):
    '''

    Finds n number of peeks in the song.
    '''

    k = (0, -2)
    c = 0
    allValues = []
    for i in array:
        k = (i, c)
        c += 1
        allValues.append(k)

    allValues = allValues.sort(key=lambda t: t[0])

    return allValues[:n]

def findRandomIndices(array, n):
    '''
    Find some random points in the song.
    '''


    import random
    setOfRandomValues = set()
    max = array.__len__()-1
    while True:
        setOfRandomValues.add(random.randint(0, max))
        if setOfRandomValues.__len__() == n:
            break
    return list(setOfRandomValues)


def findPeek(array):
    '''
    Finds the peek.
    '''

    k = (0, -2)
    c = 0
    for i in array:
        if i[0] > k[0]:
            k = (i[0], c)
        c += 1
    # print(k)
    return k


# findPeek([1,3,4,4,4,4,4])
def findPeakAndSurrounding(array, n, maxValue):
    '''
    Finds the highest peek in the song and some surrounding references.
    '''

    indices = []
    peak = findPeek(array)
    peakI = peak[1]
    for i in range(n):
        if peakI-i >= 0 and peakI+i <= maxValue:
            indices.append(peakI-i)
            indices.append(peakI+i)
    indices.append(peakI)

    return indices


def alignTwoSongs(LPSongPath, CDSongPath, numberOfReferences, searchSpace):
    '''

    This function is used to calculate and shifts the song.
    '''


    LP = wavfile.read(LPSongPath)
    CD = wavfile.read(CDSongPath)

    path = 'Training'
    path2 = 'Aligned_new'
    path = os.path.join(path, path2)
    path = os.path.join(path, CDSongPath.split('/')[-1])

    LPLength = LP.__len__()
    CDLength = CD.__len__()

    LP = LP[1]
    CD = CD[1]
    #LP = LP[1].reshape(LP[1].shape[0] * LP[1].shape[1], )
    #CD = CD[1].reshape(CD[1].shape[0] * CD[1].shape[1], )
    scaleFactor = 100000000000

    CD = CD / scaleFactor
    LP = LP / scaleFactor

    findNumber = findPeakAndSurrounding(LP[int(LP.__len__()/4):2*int((LP.__len__()/4))], int(numberOfReferences/2), CD.__len__()-searchSpace-1)#findRandomIndices(CD, 10)
    print(findNumber[-1])
    #Test1 = findPeeks(CD, 100)
    #Test2 = findPeeks(LP, 100)
    newCD = CD
    leftShift = []
    itr = searchSpace
    start = time.time()
    rightShift = []
    for i in range(0, itr, 1):
        X = []
        Y = []
        for pos in findNumber:
            x = LP[pos][0] - newCD[pos+i][0]
            X.append(abs(x))
            y = LP[pos][0] - newCD[pos - i][0]
            Y.append(abs(y))
        ySum = sum(Y)
        rightShift.append((ySum, i))
        xSum = sum(X)
        leftShift.append((xSum, i))
    print(time.time()-start)

    '''
    start = time.time()
    for i in range(itr):
        for pos in findNumber:
            if i == 0:
                x+=(LP[pos] - newCD[pos+i])
        if i == 0:
            newAttempt.append(x)
            xLast = LP[findNumber[0]]-newCD[findNumber[0]]
        else:
            x = x-xLast+(LP[findNumber[findNumber.__len__()-1]]-newCD[findNumber.__len__()-1]+1)
            xLast = LP[findNumber[0]]-newCD[findNumber[0]+i]
            newAttempt.append(x)
        newCD = np.pad(newCD, (0, 1), 'constant')
    print(time.time() - start)
    '''
    newCD = newCD * scaleFactor
    newCD = newCD.astype('int')

    rightShiftLowest = min(rightShift, key=lambda t:t[0])
    leftShiftLowest = min(leftShift, key=lambda t:t[0])


    #minsteErBest = min(joaList, key=lambda t:t[0])
    #minsteErBest = (-1, 7160)

    newCD = newCD.reshape(newCD.shape[0] * newCD.shape[1], )

    if leftShiftLowest[0] < rightShiftLowest[0]:
        minsteErBest = leftShiftLowest
        newCD = newCD[leftShiftLowest[1]*2:]
        print("left")
    else:
        print("right")
        newCD = np.pad(newCD, (rightShiftLowest[1]*2, 0), 'constant')
        minsteErBest = rightShiftLowest

    print(newCD.shape)
    newCD = newCD.reshape((int(newCD.shape[0]/2), 2))
    print(newCD.shape)

    wavfile.write(path, 96000, newCD)
    print(minsteErBest)

#alignTwoSongs('training/music_org/LPNEW(32)/LPImage [9].wav', 'training/music_org/CDNEW(32)/Image [9].wav')

def normalise(x):

    return (x + 2147483648)/(2*2147483648)

def AllignDasChunkis(path,path2,numberOfReferences, searchSpace):
    '''
    Used iterate through the list of chunks.
    '''

    for root, dirs, files in os.walk(path):
        for file in files:
            cdPath = os.path.join(root, file)
            lpPath = os.path.join(path2, "LP" + file)
            alignTwoSongs(lpPath, cdPath, numberOfReferences, searchSpace)


AllignDasChunkis('Training/Ali/CD_4.0', 'Training/Ali/LP_4.0', 3000, 9600)

def removepad(sound, before, endPos):
    '''
    Remove padding
    '''

    return sound[before:endPos]


def mergeChunks(path, savePath):

    '''
    Merged the chunks into one song.
    '''



    relatedChunks = dict()
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = file.split('_')
            if not filename[0] in relatedChunks:
                relatedChunks[filename[0]] = []
            relatedChunks[filename[0]].append(file)

    for key in relatedChunks:
        length = relatedChunks[key].__len__()
        #baseName = relatedChunks[key][0].split('_')
        sounds = []
        for i in range(1, length+1):
            #print(key)
            mergedName = key+"_"+str(i)+".wav"
            print(mergedName)
            sound = wavfile.read(os.path.join(path, mergedName))[1]
            if i == 1:
                newSound = removepad(sound, 0, 2*96000)
                soundsTest = newSound
            elif i != length+1:
                newSound = removepad(sound, 1*96000, 3*96000)
                #soundsTest = np.concatenate((soundsTest, newSound), axis=0)
            else:
                newSound = sound[:2*96000]
                #soundsTest = np.concatenate((soundsTest, newSound), axis=0)

            wavfile.write(os.path.join(savePath, mergedName), 96000, newSound)
            #print(mergedName)

            #print(type(newSound))
            #sounds +=newSound#.tolist()
        #print(sounds)
        #sounds = np.asarray(sounds)
        #newPath = os.path.join(savePath, key+".wav")
        #wavfile.write(newPath, 96000, soundsTest)


#mergeChunks('training\Aligned\CD_4.0', 'Training/Alligned chunks/CD')

def convertFrom16To32(path, toPath):
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            data, samplerate = soundfile.read(path)
            path = os.path.join(toPath, file)
            soundfile.write(path, data, samplerate, subtype='PCM_32')


#convertFrom16To32('training/music_org/CD', 'training/music_org/CDNEW(32)')

def createPictures(path, savePath):
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            sound = audiosegment.from_file(path)
            hist_bins, times, amplitudes = sound.spectrogram(window_length_s=0.03)
            hist_bins_khz = hist_bins / 1000
            amplitudes_real_normed = np.abs(amplitudes) / len(amplitudes)
            amplitudes_logged = 10 * np.log10(amplitudes_real_normed + 1e-9)  # for numerical stability
            x, y = np.mgrid[:len(times), :len(hist_bins_khz)]
            fig, ax = plt.subplots()
            ax.set_position([0, 0, 1, 1])
            ax.pcolormesh(x, y, np.swapaxes(amplitudes_logged, 0, 1))
            plt.axis('off')
            label = root.split("/")
            plt.savefig(os.path.join(savePath, file.split('.')[0] + "_" + label[len(label) - 1] + ".png"), dpi=300)
            plt.close()
    return


# createPictures('training/music_splitted/CD_1.0', 'training/pictures/CD_Pictures_1.0_300')

def griffinlim(spectrogram, n_iter=100, window='hann', n_fft=2048, hop_length=-1, verbose=False):
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
    t = tqdm.tnrange(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length=hop_length, window=window)
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length, window=window)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length=hop_length, window=window)

    return inverse


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
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
