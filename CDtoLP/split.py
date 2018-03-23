from pydub import AudioSegment
import os

#sound = AudioSegment.from_mp3("/path/to/file.mp3")




def dice(song, millisecond):
    length = int(len(song)/(millisecond))
    songs = []
    for i in range(length):
        songs.append(song[millisecond*i:millisecond*i+millisecond])
    return songs


for root, dirs, files in os.walk("LP_quality"):
    for file in files:
        path = os.path.join(root, file)
        print(path)
        print(root)
        sound = AudioSegment.from_wav(path)
        # len() and slicing are in milliseconds
        sounds = dice(sound, 100)
        i=1
        for song in sounds:
            filename = file.split(".")
            folder = os.path.basename("training")
            path = os.path.join(folder, str(filename[0] + "_" + str(i) + ".wav"))
            print(path)
            song.export(path, format="wav")
            i += 1
            #second_half_3_times.export("", format="mp3")
# Concatenation is just adding
#second_half_3_times = second_half + second_half + second_half

# writing mp3 files is a one liner
