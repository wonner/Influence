import igraph as ig
import random
import queue
import numpy as np

graph = ig.Graph.Read_Edgelist('data/slashdot.txt',True)
# 删除节点
#graph.delete_vertices(77359)
#print(graph)

# 中介中心性 太慢
#betweenness = ig.Graph.betweenness(graph,vertices=[0])
# 接近中心性 太慢
#closeness = ig.Graph.closeness(graph,mode = 1)


def networkfeature(graph):
    # 特征向量中心度
    eigen = ig.Graph.eigenvector_centrality(graph,directed=True)
    # coreness mode = OUT out-coreness out-degree
    coreness = ig.Graph.coreness(graph,mode = 1)
    # 度 out度计算，不计算self-loop
    degree = ig.Graph.degree(graph,range(77360),1,False)
    # hub score
    hubscore = ig.Graph.hub_score(graph)
    # authority score
    authorityscore = ig.Graph.authority_score(graph)
    # pagerank
    pagerank = ig.Graph.personalized_pagerank(graph)
    # Clustering Coefficient 局部集聚系数 相邻节点形成一个团的紧密程度
    clustering = ig.Graph.transitivity_local_undirected(graph)
    featurematrix = np.vstack((eigen,coreness,degree,hubscore,authorityscore,pagerank,clustering))
    return featurematrix.T

print(networkfeature(graph).shape)

# random.seed(2019)
#
# def infestimate(random,graph,id):
#     inf = 1
#     # 已激活节点标记
#     visited = {}
#     visited[id] = 1
#     # 待激活下一级节点队列
#     q = queue.Queue()
#     q.put(id)
#     while not q.empty():
#         node = q.get()
#         neighborhood = graph.neighbors(node,mode=1)
#         print(neighborhood)
#         for neighbor in neighborhood:
#             if not neighbor in visited:
#                 if random.random() > 0.5:
#                     visited[neighbor] = 1
#                     q.put(neighbor)
#                     inf = inf + 1
#     return inf
#
# inf = infestimate(random,graph,77290)
# print(inf)
