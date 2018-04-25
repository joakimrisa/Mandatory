import os
from PIL import Image
def openFile(name):
    setOfNames = set()
    with open(name, 'r') as f:
        names = f.readlines()
        for n in names:
            setOfNames.add(n.split('\n')[0].lower())


    return setOfNames

femaleNames = openFile('female.txt')
maleNames = openFile('male.txt')

#print(femaleNames)
#print(maleNames)

if not os.path.exists("gender"):
    os.makedirs(os.path.dirname("gender/"))
    os.makedirs(os.path.dirname("gender/manuel/"))
    os.makedirs(os.path.dirname("gender/female/"))
    os.makedirs(os.path.dirname("gender/male/"))

for root, dirs, files in os.walk('lfw'):
    for file in files:
        split = file.split('.')
        if not (split[1].__contains__('db')):
            name = file.split('_')
            namelower = name[0].lower()
            if femaleNames.__contains__(namelower) and not maleNames.__contains__(namelower):
                manuel = False
                newName = namelower+"_"+name[1]+"_"+"female_"+name[len(name)-1]
            elif maleNames.__contains__(namelower) and not femaleNames.__contains__(namelower):
                manuel = False
                newName = namelower+"_"+name[1]+"_"+"male_"+name[len(name)-1]
            else:
                manuel = True
                newName = namelower + "_" + name[1] + "_" + "manuel_"+name[len(name)-1]



            im = Image.open(os.path.join(root,file))
            path = 'gender'
            pathJoined = os.path.join(path, newName)
            if manuel:
                subDir = 'manuel'
                path = os.path.join(path, subDir)
                im.save(os.path.join(path, newName))
            else:
                im.save(pathJoined)
            #Y.append(name[0] + name[1])

