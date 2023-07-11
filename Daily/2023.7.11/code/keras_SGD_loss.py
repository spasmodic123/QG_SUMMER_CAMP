from keras.models import Sequential
from keras.layers import Activation,Dense
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras import datasets
from keras import utils

# 观察数据的维度,判断输入和输出节点数量
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)  60000个样本,28*28个特征,也就是维度,输入节点28*28个
print(y_train.shape)  # (60000,)  输出只有维度是1,一个输出,一个输出节点

x_train = x_train[:10000]  # 数据量太大,只选其中10000个
y_train = y_train[:10000]

# 将数据reshape并缩放
x_train = x_train.reshape((-1, 28 * 28))
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((-1,28*28))
x_test = x_test.astype('float32') / 255

# 转化为onehot编码
y_train = utils.to_categorical(y_train,num_classes= 10)
y_test = utils.to_categorical(y_test,num_classes= 10)

print('------------------')
print(x_train.shape)
print(y_train.shape)  # (10000,10)  onehot编码之后输出维度变成10,一个输出,十个输出节点

# 建立神经网络模型
model = Sequential([
    Dense(units=512,input_shape=(28*28,),activation='relu'),
    Dense(units=10,activation='softmax')
])

model.compile(  # compile()传递了优化器(优化函数),损失函数,以及我们想要看到的指标
    optimizer=Adam(learning_rate=0.0001),  # Adam算法是一种随机梯度下降算法
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

model.summary()  # 概括模型

model.fit(
    x=x_train,
    y=y_train,
    batch_size=10,  # 一次向模型发送10个样本
    epochs=20,  # 全部数据经过模型20次
    shuffle=True,  # 数据洗牌,打乱数据
    verbose=1  # 日志记录模式      0 = 安静模式, 1 = 进度条, 2 = 每轮一行
)

# 模型预测
prediction = model.predict(
    x=x_test,
    batch_size=10,
    verbose=1
)
for i in range(5):
    print(prediction[i])
# [1.5765100e-07 2.2723780e-10 1.5910067e-05 3.6559248e-04 3.9271711e-10
#  1.7562368e-07 9.5145170e-13 9.9961543e-01 4.0251831e-07 2.3608497e-06]
# [7.7392264e-07 6.8146001e-05 9.9970311e-01 1.6512492e-04 8.7406211e-14
#  1.8420375e-05 4.2345957e-05 3.2205594e-10 2.1628621e-06 1.7322250e-11]
# [8.2852063e-07 9.9483478e-01 2.7394507e-03 3.8868238e-04 1.0728442e-04
#  5.5759789e-05 7.0825648e-05 6.6760078e-04 1.1303802e-03 4.4597255e-06]
# [9.9997962e-01 2.1234193e-11 8.0121854e-06 9.0892179e-08 2.9640925e-12
#  2.4460746e-07 2.1883823e-06 7.7016803e-06 4.5301687e-09 2.1758046e-06]
# [1.8581100e-06 3.5931970e-07 4.7599631e-05 8.5706620e-07 9.8842013e-01
#  8.8175369e-07 8.3759605e-06 5.4352346e-05 3.4691609e-05 1.1430876e-02]
'''输出前五个预测数据,每一个预测输出都是一个列表,列表有10列,因为我们输出结点为10个,也就是结果可以分为10个类别
每一个预测输出列表中的10个值加起来=1,就是说预测本质上就是概率的预测,
每个预测输出的每个数值,代表最终预测结果是该类别的概率'''