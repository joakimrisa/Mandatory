import os
import re


def forFolderWords(path, n=10):
    '''

    This function loads in articles according to words
    '''

    data = []
    c = 0
    nonWords = set()
    nonWords.add(' ')
    nonWords.add('')
    for root, dirs, files in os.walk(path):
        for file in files:
            lines = open(os.path.join(root, file), errors='ignore').readlines()
            for word in lines:
                words = word.split(' ')
                for w in words:

                    w = re.sub('[^A-Za-z0-9]+', '', w)
                    if not nonWords.__contains__(w):
                        data.append(w.lower())

            c += 1
            if n == c:
                chars = sorted(list(set(data)))
                totalChars = len(data)
                numberOfUniqueChars = len(chars)
                CharsForids = {char: Id for Id, char in enumerate(chars)}
                idsForChars = {Id: char for Id, char in enumerate(chars)}
                numberOfCharsToLearn = 100
                counter = totalChars - numberOfCharsToLearn
                charX = []
                y = []
                for i in range(0, counter):
                    theInputChars = data[i:i + numberOfCharsToLearn]
                    theOutputChars = data[i + numberOfCharsToLearn]
                    charX.append([CharsForids[char] for char in theInputChars])
                    y.append(CharsForids[theOutputChars])
                return (charX, y, numberOfCharsToLearn, numberOfUniqueChars, idsForChars)


def forFolder(path, n=10):
    '''
    This folder loads in articles according to chars
    '''
    data = []
    c = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            data += open(os.path.join(root, file), errors='ignore').read().lower()
            c += 1
            if n == c:
                print(file)
                chars = sorted(list(set(data)))
                totalChars = len(data)
                numberOfUniqueChars = len(chars)
                CharsForids = {char: Id for Id, char in enumerate(chars)}
                idsForChars = {Id: char for Id, char in enumerate(chars)}
                numberOfCharsToLearn = 100
                counter = totalChars - numberOfCharsToLearn
                charX = []
                y = []

                for i in range(0, counter, 1):
                    theInputChars = data[i:i + numberOfCharsToLearn]
                    theOutputChars = data[i + numberOfCharsToLearn]
                    charX.append(([CharsForids[char] for char in theInputChars]))
                    y.append(CharsForids[theOutputChars])

                return (charX, y, numberOfCharsToLearn, numberOfUniqueChars, idsForChars)


def preprocessing(name):
    '''
    This function loads in an article and preprocesses it for char networks
    '''
    data = open(name, errors='ignore').read().lower()
    chars = sorted(list(set(data)))
    totalChars = len(data)
    numberOfUniqueChars = len(chars)
    CharsForids = {char: Id for Id, char in enumerate(chars)}
    idsForChars = {Id: char for Id, char in enumerate(chars)}
    numberOfCharsToLearn = 100
    counter = totalChars - numberOfCharsToLearn
    charX = []

    y = []

    for i in range(0, counter, 1):
        theInputChars = data[i:i + numberOfCharsToLearn]
        theOutputChars = data[i + numberOfCharsToLearn]
        charX.append([CharsForids[char] for char in theInputChars])
        y.append(CharsForids[theOutputChars])

    return (charX, y, numberOfCharsToLearn, numberOfUniqueChars, idsForChars)
