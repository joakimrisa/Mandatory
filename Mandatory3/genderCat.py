import os
from PIL import Image
def openFile(name):
    '''
    This function takes in a list of common female and male names and goes through the dataset to distinguish females from males.
    '''
    setOfNames = set()
    with open(name, 'r') as f:
        names = f.readlines()
        for n in names:
            setOfNames.add(n.split('\n')[0])


    return setOfNames

femaleNames = openFile('femaleR.txt')
maleNames = openFile('maleR.txt')

if not os.path.exists("gender"):
    os.makedirs(os.path.dirname("gender/"))
    os.makedirs(os.path.dirname("gender/female/"))
    os.makedirs(os.path.dirname("gender/manual/"))
    os.makedirs(os.path.dirname("gender/male/"))

for root, dirs, files in os.walk('lfw'):
    for file in files:
        split = file.split('.')
        if not (split[1].__contains__('db')):
            name = file.split('_')
            namelower = name[0].lower()
            im = Image.open(os.path.join(root,file))
            path = 'gender'
            if femaleNames.__contains__(file):
                newName = namelower + "_" + name[1] + "_" + "female_" + name[len(name) - 1]
                subDir = 'female'
                path = os.path.join(path, subDir)
                im.save(os.path.join(path, newName))
            elif maleNames.__contains__(file):
                newName = namelower + "_" + name[1] + "_" + "male_" + name[len(name) - 1]
                subDir = 'male'
                path = os.path.join(path, subDir)
                im.save(os.path.join(path, newName))
            else:
                newName = namelower + "_" + name[1] + "_" + "manual_"+name[len(name)-1]
                subDir = 'manual'
                path = os.path.join(path, subDir)
                im.save(os.path.join(path, newName))


