from keras.models import Sequential
from keras.layers import Dense,Activation

layers = [
    Dense(units=6,input_shape=(8,), activation='relu'),  # units是输出的维度,input_shape代表输入层有两个节点,也就输输入维度是2,也就是两个特征,activation是激活函数
    Dense(units=6, activation='relu'),  # Dense是一种典型的层,称为全连接层或密集层,意思是该层的每一个节点连接上一层的所有节点和下一层的所有节点,深度学习还有很多其他层,卷积层就是另外一种
    Dense(units=4, activation='softmax'),
]
model1 = Sequential(layers)  # Sequential是一个顺序模型(序列模型),是线性层的序列堆栈,正如神经网络按层组织
model1.add(Dense(units=3,activation='relu'))
# 8个输入,6个输出到第二层
# 6个输入到第二层,6个输出给第三层
# 6个输入到第三层,4个输出到第四层
# 4个输入到第四层,3个输入到第五层