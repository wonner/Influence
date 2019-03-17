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


