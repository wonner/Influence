import keras
from keras.datasets import boston_housing
from keras import models
from keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import igraph_feature
import networkx as nx
import igraph as ig
import numpy as np

def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# 种子数量
k=2
graph = ig.Graph.Read_Edgelist('data/karate.txt',True)
reverseGraph = ig.Graph.Read_Edgelist('data/karate-reverse.txt',True)
graph.delete_vertices(0)
directGraph = nx.read_adjlist('data/karate.txt',create_using=nx.DiGraph())

embedding, (sample_data, sample_targets, sample_id, prediction, prediction_id) = igraph_feature.load_data(graph,reverseGraph,directGraph,0.9)
margin = int(len(sample_targets)*0.1)
test_data = sample_data[0:margin,]
test_targets = sample_targets[0:margin]
train_data = sample_data[margin:len(sample_targets),]
train_targets = sample_targets[margin:len(sample_targets)]

# 参数设置
batch_size = 27
num_epoch = 100
validation_split = 0.1

# 正则化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

model = build_model()
history = model.fit(train_data,test_data,batch_size,num_epoch,validation_split)
mae_history = history.history['val_mean_absolute_error']

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)

# 绘制验证误差随轮次变化曲线
plt.plot(range(1, len(mae_history) + 1), mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.savefig('data/curve.png')

predict = model.predict(prediction)

# 剩余节点id
remain = list(range(graph.vcount()))
# 选取第一个种子节点
seeds = []
inf = 0
seed = 0
for index,val in enumerate(sample_targets):
    if val>inf:
        seed = sample_id[index]
        inf = val
for index,val in enumerate(predict):
    if val>inf:
        seed = prediction_id[index]
        inf = val
seeds.append(seed)

# 选取剩余种子节点
while len(seeds) != k:
    graph.delete_vertices(seed)
    reverseGraph.delete_vertices(seed)
    directGraph.remove_node(remain[seed])
    remain.remove(seed)

    feature = []
    feature.append(igraph_feature.networkfeature(graph,reverseGraph))
    feature.append(embedding.learn_embedding(directGraph))
    feature -= mean
    feature /= std
    predict = model.predict(np.array(feature))
    inf = 0
    seed = 0
    for index,val in enumerate(predict):
        if val>inf:
            seed = index
            inf = val
    seeds.append(remain[seed])

print(seeds)
