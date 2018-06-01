import random
import codecs
import csv
import math
import pickle
import sys
import time

MAXPHEROMONES = 100000
MINPHEROMONES = 1
nodes = dict()
edges = dict()
clusterNodes = dict()
MAXCOST = 0
bestScore = 0
bestSolution = []
currentScore = 0


class clusterNode:
    def __init__(self, name,):
        self.name = name
        self.nodes = dict()

class Node:
    def __init__(self, name, ):
        self.name = name
        self.edges = []

    def rouletteWheelSimple(self):
        return random.sample(self.edges, 1)[0]

    def rouletteWheel(self, visitedEdges, startNode, endNode, numCities):
        visitedNodes = [oneEdge.toNode for oneEdge in visitedEdges]
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
        while s <= num:
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


class City:
    def __init__(self, country, name, population, la, lo):
        self.country = country
        self.name = name
        self.population = population
        self.la = la
        self.lo = la


class Greedy:
    def __init__(self):
        self.visitedEdges = []
        self.visitedNodes = []

    def walk(self, startNode):
        currentNode = startNode
        currentEdge = None
        while (not checkAllNodesPresent(self.visitedEdges)):
            possibleEdges = [(edge.cost, edge) for edge in currentNode.edges if edge.toNode not in self.visitedNodes]
            possibleEdges.sort(key=lambda edge: edge[0])
            currentEdge = possibleEdges[0][1]
            currentNode = currentEdge.toNode
            self.visitedEdges.append(currentEdge)
            self.visitedNodes.append(currentNode)
            print(currentNode, currentEdge)


def loader2(country="no", limit=100):
    global nodes
    global edges
    name = country + "_" + str(limit)
    sys.setrecursionlimit(limit*10)
    try:
        nodes = pickle.load(open(name + "_nodes.p", "rb"))
        edges = pickle.load(open(name + "_edges.p", "rb"))
    except:
        print("Shouldn't load edges and nodes...")
        with codecs.open("worldcitiespop.csv", "r", encoding="utf-8", errors="ignore") as f:
            allCities = csv.reader(f, delimiter=',', quotechar='|')
            uniqueCities = []
            usedCities = set()
            usedLocations = set()
            for city in allCities:
                if city[0] == country:
                    la = int(float(city[5]) * 100)
                    lo = int(float(city[6]) * 100)
                    if uniqueCities.__len__() == limit:
                        break
                    if not usedCities.__contains__(city[1]) and not usedLocations.__contains__((la, lo)) and city[
                        1].__len__() > 2:
                        uniqueCities.append(city)
                        usedLocations.add((la, lo))
                        usedCities.add(city[1])
                        nodes[city[1]] = Node(city[1])

        for departure in uniqueCities:
            la1 = departure[5]  # x
            lo1 = departure[6]  # y
            for destination in uniqueCities:
                if departure != destination:
                    la2 = destination[5]  # x1
                    lo2 = destination[6]  # y1
                    xDot = abs(float(la2) - float(la1))
                    yDot = abs(float(lo2) - float(lo1))
                    distance = math.sqrt(xDot ** 2 + yDot ** 2)
                    if not departure[1] in edges:
                        edges[departure[1]] = []
                    edges[departure[1]].append(
                        Edge(nodes[departure[1]], nodes[destination[1]], distance))
            nodes[departure[1]].edges = edges[departure[1]]
        pickle.dump(nodes, open(name + "_nodes.p", "wb"))
        pickle.dump(edges, open(name + "_edges.p", "wb"))


def loader(country="no", limit=100):
    '''
    This function loads in the cities from a given country and a defined limit
    '''
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
                la = int(float(line[5]) * 100)
                lo = int(float(line[6]) * 100)
                hashval = la * lo
                if not hashval in possibrahCities:
                    if not line[1] in nodes:
                        possibrahCities[hashval] = line
                        nodes[line[1]] = Node(line[1].upper())

                elif possibrahCities[hashval][4] < line[4]:
                    del nodes[possibrahCities[hashval][1]]
                    possibrahCities[hashval] = line
                    nodes[line[1]] = Node(line[1].upper())


                elif possibrahCities[hashval][1] != line[1]:
                    del nodes[possibrahCities[hashval][1]]
                    possibrahCities[hashval] = line
                    nodes[possibrahCities[hashval][1]] = Node(line[1].upper())

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
                if not possibrahCities[departure][1] in edges:
                    edges[possibrahCities[departure][1]] = []
                edges[possibrahCities[departure][1]].append(
                    Edge(nodes[possibrahCities[departure][1]], nodes[possibrahCities[destination][1]], distance))

        nodes[possibrahCities[departure][1]].edges = edges[possibrahCities[departure][1]]
        edges[possibrahCities[departure][1]] = []
        print(nodes[possibrahCities[departure][1]].edges.__len__())
    print(nodes.__len__())



def getSum(edges):
    return sum(e.cost for e in edges)


class ANT:
    def __init__(self):
        self.visitedEdges = []

    def walk(self, startNode, endNode, numCities):
        currentNode = startNode
        currentEdge = None
        while (not self.visitedEdges.__len__() >= numCities + 1):
            currentEdge = currentNode.rouletteWheel(self.visitedEdges, startNode, endNode, numCities)
            currentNode = currentEdge.toNode
            self.visitedEdges.append(currentEdge)

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
        edge.pheromones *= 0.95


def checkAllEdges(edges):
    for edge in edges:
        edge.checkPheromones()

def multiLayer(clusterSize = 10):
    '''
    This function defines the clustering
    '''
    global nodes
    global edges
    global clusterNodes
    lastClusterTown = None
    alreadyClustered = set()
    for node in nodes:

        if not alreadyClustered.__contains__(node):
            clusterNodes[node] = clusterNode(node)
            clusterNodes[node].nodes[nodes[node].name] = nodes[node]
            alreadyClustered.add(node)
            while clusterNodes[node].nodes.__len__() < clusterSize:
                minCost = 100
                minEdge = None
                for edge in edges[node]:
                    if edge.cost < minCost:
                        if not alreadyClustered.__contains__(edge.toNode.name):
                            minCost = edge.cost
                            minEdge = edge
                if minEdge != None:
                    clusterNodes[node].nodes[minEdge.toNode.name] = nodes[minEdge.toNode.name]
                    alreadyClustered.add(minEdge.toNode.name)
                    edges[node].remove(minEdge)
    for cluster in clusterNodes:
        for node in clusterNodes[cluster].nodes:
            edgeList = []
            for edge in clusterNodes[cluster].nodes[node].edges:
                if edge.toNode.name in clusterNodes[cluster].nodes:
                    edgeList.append(edge)
                elif edge.toNode.name in clusterNodes:
                    edgeList.append(edge)
            clusterNodes[cluster].nodes[node].edges = edgeList

def findCity(city):
    '''
    This checks which cluster an input city is in.
    '''
    global clusterNodes
    for cluster in clusterNodes:
        if city in clusterNodes[cluster].nodes:
            return cluster

def runWalk2(startCity, endCity, numCity=10, iterations=10):
    '''
    This function uses ant with pheromones combined with clustering to find the best route from a startcity and endcity
     with number of Cities defined and iterations.
    '''

    global MAXCOST
    global clusterNodes
    last = 0
    counter = 0
    allSums = set()
    for city in edges:
        MAXCOST += getSum(edges[city])
    #currentSum = 0
    startCluster = findCity(startCity)
    endCluster = findCity(endCity)
    while counter < iterations:
        for cluster in clusterNodes:
            for city in clusterNodes[cluster].nodes:
                evaporate(clusterNodes[cluster].nodes[city].edges)
                ant = ANT()
                ant.walk(clusterNodes[startCluster].nodes[startCity], clusterNodes[endCluster].nodes[endCity], numCity)
                ant.pheromones()
                checkAllEdges(clusterNodes[cluster].nodes[city].edges)
                last = currentScore
                currentSum = getSum(ant.visitedEdges)
                allSums.add(currentSum)
                if counter > int(0.9*iterations):
                    print(currentSum)
            counter += 1
    print(currentSum)
    print(min(allSums))
    print(ant.visitedEdges)

def runWalk(startCity, endCity, numCity=10, iterations=10):
    '''
    This function uses ant with pheromones to find the best route from a startcity and endcity
     with number of Cities defined and iterations.
    '''
    global MAXCOST
    last = 0
    counter = 0
    allSums = set()
    for city in edges:
        MAXCOST += getSum(edges[city])
    while counter < iterations:
        for city in nodes:
            evaporate(nodes[city].edges)
            ant = ANT()
            ant.walk(nodes[startCity], nodes[endCity], numCity)
            ant.pheromones()
            checkAllEdges(nodes[city].edges)
            last = currentScore
            currentSum = getSum(ant.visitedEdges)
            allSums.add(currentSum)
            if counter > int(0.9 * iterations):
                print(currentSum)
        counter += 1
    print(currentSum)
    print(min(allSums))
    print(ant.visitedEdges)


loader2('no', 1000)

'''
Runs the different solution where the goal is to get from aabjorgan to aadalsbruk with 25 cities in between
'''

start = time.time()
runWalk('aabjorgan', 'aadalsbruk', 25, iterations=5000)
end = time.time()
print(end-start)

multiLayer(50)
start = time.time()
runWalk2('aabjorgan', 'aadalsbruk', 25, iterations=5000)
end = time.time()
print(end-start)
