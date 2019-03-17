import sys
import os
from time import time
import networkx as nx
sys.path.append('/Users/yangwan/Desktop/code/digit/DynamicGEM')

from DynamicGEM.dynamicgem.embedding.ae_static    import AE

file = open('output/feature.txt')
# 读入有向图
directGraph = nx.read_adjlist('/Users/yangwan/Desktop/code/digit/data/slashdot1.txt',create_using=nx.DiGraph())

# 向量维数
dim_emb = 128
embedding = AE(d            = dim_emb,
               beta       = 5,
               nu1        = 1e-6,
               nu2        = 1e-6,
               K          = 3,
               n_units    = [500, 300, ],
               n_iter     = 200,
               xeta       = 1e-4,
               n_batch    = 100,
               modelfile  = ['./intermediate/enc_modelsbm.json',
                             './intermediate/dec_modelsbm.json'],
               weightfile = ['./intermediate/enc_weightssbm.hdf5',
                             './intermediate/dec_weightssbm.hdf5'])

emb,_ = embedding.learn_embeddings(directGraph)

file.write(emb)

