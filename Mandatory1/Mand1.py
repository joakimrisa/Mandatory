import random
import codecs
import csv
import math

MAXPHEROMONES = 100000
MINPHEROMONES = 1
nodes = dict()
edges = dict()
MAXCOST = 0
bestScore = 0
bestSolution = []
currentScore = 0


class Node:
    def __init__(self, name):
        self.name = name
        self.edges = []

    def rouletteWheelSimple(self):
        return random.sample(self.edges, 1)[0]

    def rouletteWheel(self, visitedEdges, startNode, endNode, numCities):
        visitedNodes = [oneEdge.toNode for oneEdge in visitedEdges]
        # print(visitedNodes.__len__())
        if numCities <= visitedNodes.__len__():
            for edge in self.edges:
                if edge.toNode == endNode:
                    viableEdges = [edge]
                    break
        else:
            viableEdges = [oneEdge for oneEdge in self.edges if
                           not oneEdge.toNode in visitedNodes and oneEdge.toNode != startNode and oneEdge.toNode != endNode]

        allPheromones = sum([oneEdge.pheromones for oneEdge in viableEdges])
        num = random.uniform(0, allPheromones)
        s = 0
        i = 0

        selectedEdge = viableEdges[i]
        while (s <= num):
            selectedEdge = viableEdges[i]
            s += selectedEdge.pheromones
            i += 1
        return selectedEdge

    def __repr__(self):
        return self.name


class Edge:
    def __init__(self, fromNode, toNode, cost):
        self.fromNode = fromNode
        self.toNode = toNode
        self.cost = cost
        self.pheromones = MAXPHEROMONES

    def checkPheromones(self):
        if (self.pheromones > MAXPHEROMONES):
            self.pheromones = MAXPHEROMONES
        if (self.pheromones < MINPHEROMONES):
            self.pheromones = MINPHEROMONES

    def __repr__(self):
        return self.fromNode.name + "--(" + str(self.cost) + ")--" + self.toNode.name

def checkAllNodesPresent(edges):
    visitedNodes = [edge.toNode for edge in edges]

    return set(nodes.values()).issubset(visitedNodes)


class Greedy:
    def __init__(self):
        self.visitedEdges = []
        self.visitedNodes = []

    def walk(self, startNode):
        currentNode = startNode
        currentEdge = None
        while (not checkAllNodesPresent(self.visitedEdges)):
            possibleEdges = [(edge.cost, edge) for edge in currentNode.edges if edge.toNode not in self.visitedNodes]
            # print(possibleEdges)
            # possibleEdges.sort()
            possibleEdges.sort(key=lambda edge: edge[0])
            # import pdb;pdb.set_trace()
            currentEdge = possibleEdges[0][1]
            currentNode = currentEdge.toNode
            self.visitedEdges.append(currentEdge)
            self.visitedNodes.append(currentNode)
            print(currentNode, currentEdge)


def loader(country="no", limit=100):
    # Make so it asks for country file
    global edges
    global nodes
    with codecs.open("worldcitiespop.csv", "r", encoding="utf-8", errors="ignore") as f:
        allCities = csv.reader(f, delimiter=',', quotechar='|')
        possibrahCities = dict()
        for line in allCities:
            if possibrahCities.__len__() == limit:
                break
            if line[0] == country and line[1].__len__() > 2:
                line[1] = line[1].replace(" ", "")
                print("Starter på nytt dd")
                # nodes[line[1]] = Node(line[1].upper())
                la = int(float(line[5]) * 100)
                lo = int(float(line[6]) * 100)
                hashval = la * lo
                if not hashval in possibrahCities:
                    if not line[1] in nodes:
                        print(len(possibrahCities))
                        print(len(nodes))
                        possibrahCities[hashval] = line
                        nodes[line[1]] = Node(line[1].upper())
                        print("1")
                        print(len(possibrahCities))
                        print(len(nodes))

                elif possibrahCities[hashval][4] < line[4]:
                    del nodes[possibrahCities[hashval][1]]
                    possibrahCities[hashval] = line
                    nodes[line[1]] = Node(line[1].upper())
                    print("2")

                elif possibrahCities[hashval][1] != line[1]:
                    print("3")
                    print(len(possibrahCities))
                    print(len(nodes))
                    del nodes[possibrahCities[hashval][1]]
                    possibrahCities[hashval] = line
                    print(len(line[1]))
                    if line[1] == 'akarvik':
                        print("EG E NÅ UTEN SPACE")
                    print(line[1])
                    nodes[possibrahCities[hashval][1]] = Node(line[1].upper())

                #kebabfix, må endres om goody sier at vi må ha alle byer med ;)

    for i in possibrahCities:
        if not possibrahCities[i][1] in nodes:
            print("AVIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIK")

    print(possibrahCities)
    print(nodes)


    for departure in possibrahCities:
        la1 = possibrahCities[departure][5]  # x
        lo1 = possibrahCities[departure][6]  # y
        for destination in possibrahCities:
            if departure != destination:
                la2 = possibrahCities[destination][5]  # x1
                lo2 = possibrahCities[destination][6]  # y1
                xDot = abs(float(la2) - float(la1))
                yDot = abs(float(lo2) - float(lo1))
                distance = math.sqrt(xDot ** 2 + yDot ** 2)
                # print(distance)
                ##
                if not possibrahCities[departure][1] in edges:
                    edges[possibrahCities[departure][1]] = []
                edges[possibrahCities[departure][1]].append(
                    Edge(nodes[possibrahCities[departure][1]], nodes[possibrahCities[destination][1]], distance))

        nodes[possibrahCities[departure][1]].edges = edges[possibrahCities[departure][1]]
        edges[possibrahCities[departure][1]] = []
        print(nodes[possibrahCities[departure][1]].edges.__len__())
    # print(nodes['aabjorgan'].edges)
    print(nodes.__len__())
    # print(nodes['aabjorgan'].edges.__len__())
    # Write to file


'''
print("Greedy")
g = Greedy()
g.walk(a)
print("Cost:", sum([e.cost for e in g.visitedEdges]))


# Cost function
'''


def getSum(edges):
    return sum(e.cost for e in edges)


class ANT:
    def __init__(self):
        self.visitedEdges = []

    def walk(self, startNode, endNode, numCities):
        currentNode = startNode
        # print(type(currentNode))
        currentEdge = None
        while (not self.visitedEdges.__len__() >= numCities + 1):
            # print(self.visitedEdges)
            currentEdge = currentNode.rouletteWheel(self.visitedEdges, startNode, endNode, numCities)
            # print(type(currentEdge))
            currentNode = currentEdge.toNode
            self.visitedEdges.append(currentEdge)

        ''' 
        #print(self.visitedEdges)
      for edge in edges[currentNode.name.lower()]:
            if(edge.toNode == endNode):
                currentEdge = edge
                self.visitedEdges.append(currentEdge)
                print("duja")
                break
'''

    def pheromones(self):
        global MAXCOST
        currentCost = getSum(self.visitedEdges)
        if (currentCost < MAXCOST):
            score = 1000 ** (1 - float(currentCost) / MAXCOST)  # Score function
            global bestScore
            global bestEdges
            global currentScore
            currentScore = score
            if (score > bestScore):
                bestScore = score
                bestEdges = self.visitedEdges
            for oneEdge in bestEdges:
                oneEdge.pheromones += score


def evaporate(edges):
    for edge in edges:
        edge.pheromones *= 0.99


def checkAllEdges(edges):
    for edge in edges:
        edge.checkPheromones()


def runWalk(startCity, endCity, numCity=10, threshold=10):
    global MAXCOST
    last = 0
    counter = 0
    allSums = set()
    for city in edges:
        MAXCOST += getSum(edges[city])
    while True:
        if last == bestScore:
            counter += 1
            if counter == threshold:
                # print("hola")
                break
        else:
            # print("else")
            counter = 0
        for city in nodes:
            evaporate(nodes[city].edges)
            ant = ANT()
            # print(type(nodes['aabjorgan']))
            ant.walk(nodes[startCity], nodes[endCity], numCity)
            ant.pheromones()
            checkAllEdges(nodes[city].edges)
            last = currentScore
            # print i,getSum(ant.visitedEdges)
            currentSum = getSum(ant.visitedEdges)
            allSums.add(currentSum)
    print(currentSum)
    print(min(allSums))
    # print(ant.visitedEdges)


loader('no', 1000)
runWalk('aabjorgan', 'aadalsbruk', 10, threshold=30)

'''
for i in range(100000):
    evaporate(edges)
    ant = ANT()
    ant.walk(a)
    ant.pheromones()
    checkAllEdges(edges)
    # print i,getSum(ant.visitedEdges)
    print(getSum(ant.visitedEdges))

# Printing
ant = ANT()
ant.walk(a)
for edge in ant.visitedEdges:
    print(edge, edge.pheromones)
'''
