def toNumber(name):

    if name == 'male':
        return 0
    elif name == 'female':
        return 1
    elif name == 'manuel':
        return 2

def fromNumber(n):

    if n == 0:
        return 'male'
    elif n == 1:
        return 'female'
    elif n == 2:
        return 'manuel'