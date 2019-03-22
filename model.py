import keras
from keras.datasets import boston_housing
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import igraph_feature


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


embedding, (train_data, train_targets, prediction) = igraph_feature.load_data('data/slashdot1.txt',0.1)
margin = int(len(train_targets)*0.1)
test_data = train_data[0:margin,]
test_targets = train_targets[0:margin]
train_data = train_data[margin:len(train_targets),]
train_targets = train_targets[margin:len(train_targets)]

# 参数设置
batch_size = 128
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
plt.show()

predict = model.predict(prediction)
