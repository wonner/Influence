import igraph as ig
import random
import queue
import numpy as np
import sys

sys.path.append('/Users/yangwan/Desktop/code/digit/DynamicGEM')
from DynamicGEM.dynamicgem.embedding.ae_static    import AE

# 删除节点
#graph.delete_vertices(77359)
#print(graph)

# 中介中心性 太慢
#betweenness = ig.Graph.betweenness(graph,vertices=[0])
# 接近中心性 太慢
#closeness = ig.Graph.closeness(graph,mode = 1)


def networkfeature(graph,reverseGraph):
    # 特征向量中心度
    eigen = ig.Graph.eigenvector_centrality(graph,directed=True)
    # coreness mode = OUT out-coreness out-degree
    coreness = ig.Graph.coreness(graph,mode = 1)
    # 度 out度计算，不计算self-loop
    degree = ig.Graph.degree(graph,range(graph.vcount()),1,False)
    # hub score
    hubscore = ig.Graph.hub_score(reverseGraph)
    # authority score
    authorityscore = ig.Graph.authority_score(reverseGraph)
    # pagerank
    pagerank = reverseGraph.pagerank()
    # Clustering Coefficient 局部集聚系数 相邻节点形成一个团的紧密程度
    clustering = ig.Graph.transitivity_local_undirected(graph)
    featurematrix = np.vstack((eigen,coreness,degree,hubscore,authorityscore,pagerank,clustering))
    return featurematrix.T

#random.seed(2019)

def infestimate(random,graph,id):
    inf = 1
    # 已激活节点标记
    visited = {}
    visited[id] = 1
    # 待激活下一级节点队列
    q = queue.Queue()
    q.put(id)
    while not q.empty():
        node = q.get()
        neighborhood = graph.neighbors(node,mode=1)
        print(neighborhood)
        for neighbor in neighborhood:
            if not neighbor in visited:
                if random.random() > 0.5:
                    visited[neighbor] = 1
                    q.put(neighbor)
                    inf = inf + 1
    return inf

# inf = infestimate(random,graph,7446)
# print(inf)


def load_data(graph,reverseGraph,directGraph,trainpercent=0.1):
    random.seed(2019)
    # 计算特征
    igraphfeature = networkfeature(graph,reverseGraph)
    dim_emb = 4
    embedding = AE(d            = dim_emb,
               beta       = 5,
               nu1        = 1e-6,
               nu2        = 1e-6,
               K          = 3,
               n_units    = [500, 300, ],
               n_iter     = 30,
               xeta       = 1e-4,
               n_batch    = 30,
               modelfile  = ['data/enc_modelsbm.json',
                             'data/dec_modelsbm.json'],
               weightfile = ['data/enc_weightssbm.hdf5',
                             'data/dec_weightssbm.hdf5'])

    emb,_ = embedding.learn_embeddings(directGraph)
    feature = np.hstack((igraphfeature,emb))
    # 抽取部分点作为训练集
    vertexnumber = graph.vcount()
    traindata = []
    traintarget = []
    nodeid = random.sample(range(vertexnumber),int(vertexnumber*trainpercent))
    #
    nodeid.sort()
    for i in nodeid:
        inf=infestimate(random,graph,i)
        traindata.append(feature[i])
        traintarget.append(inf)
    # 删除训练集节点特征
    feature = np.delete(feature,nodeid,0)
    # 反向删除训练集节点索引
    remain = list(range(vertexnumber))
    for i in nodeid[::-1]:
        remain.remove(remain[i])

    return embedding, (np.array(traindata), np.array(traintarget), nodeid, feature, remain)

