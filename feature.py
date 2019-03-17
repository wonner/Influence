import numpy as np


class Graph:

    def __init__(self,graph=None):
        if graph is None:
            graph = {}
        self.graph = graph

    # 添加顶点
    def addVertex(self,vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    # 添加边
    def addEdge(self,fromvertex,tovertex):
        if fromvertex in self.graph:
            self.graph[fromvertex].append(tovertex)
        else:
            self.graph[fromvertex] = [tovertex]

    # 顶点度
    def degree(self,vertex):
        if vertex in self.graph:
            return len(self.graph[vertex])
        else:
            return 0

    # test
    def getVertex(self):
        return list(self.graph.keys())

    def getedge(self,vertex):
        return self.graph[vertex]


# 计算最大节点度
def maxdegree(graph,vertexnum):
    maxdegree = 0
    for x in range(vertexnum):
        degree = graph.degree(str(x))
        if degree > maxdegree:
            maxdegree = degree
    return maxdegree

# 点度中心性
def degreecentrality(graph,vertex,maxdegree):
    return graph.degree(vertex)/maxdegree

# 生成邻接矩阵
def generateMatrix(graph,dimension):
    A = np.zeros((dimension,dimension))
    for vertex in range(dimension):
        if str(vertex) in graph.graph:
            list = graph.graph[str(vertex)]
            for i in list:
                A[vertex,int(i)] = 1
    return A

# 特征向量中心性
def powerIteration(A):
    (row, col) = A.shape
    assert(row == col)
    dimension = row
    v = np.random.rand(dimension)
    counter = 0
    iteration_max = 1
    eps = 1.0e-15
    while(counter < iteration_max):
        counter += 1
        v_updated = A.dot(v)
        v_updated = v_updated/np.linalg.norm(v_updated)
        error = np.linalg.norm(v - v_updated)
        print ("counter = " + str(counter) + ", error = " + str(error))
        if (error < eps):
            break
        v = v_updated
    eigenvalue = v_updated.dot(A).dot(v_updated)/v_updated.dot(v_updated)
    return (eigenvalue, v_updated)




# read graph
graph = Graph()
with open('data/slashdot.txt','r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        else:
            line = line.split()
            graph.addVertex(line[0])
            graph.addEdge(line[0],line[1])

#print(graph.degree('7'))
#print(maxdegree(graph,77360))
#print(degreecentrality(graph,'7',2508))
(eigenvalue, v_updated) = powerIteration(generateMatrix(graph,77360))
print(eigenvalue)
print(len(v_updated))


