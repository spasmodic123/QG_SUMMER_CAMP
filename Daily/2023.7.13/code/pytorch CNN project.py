import torch
import torch.nn as nn  # 子包,包括构建神经网络的模块和可拓展类
import torch.optim as optim  # 包括标准化操作,SGD,Adam等
import torch.nn.functional as F  # 函数接口,包括构建神经网络的典型操作,比如损失函数和卷积
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms  # 用于图像处理的常用变换

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import pdb
import os

torch.set_grad_enabled(True)

# ETL 过程,简称1.抓取数据,2.转换数据和3.加载数据的过程
torch.set_printoptions(linewidth=120)

'''# 声明用GPU跑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'''

train_set = torchvision.datasets.FashionMNIST(
    root='.\data'
    , train=True
    , download=True
    , transform=transforms.Compose([transforms.ToTensor()])  # 转化为tensor形式
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000,
                                           shuffle=True)  # 数据加载器,用于加载数据,将数据放在特定的数据结构中,让机器易于访问
batch = next(iter(train_loader))  # 将数据解压为两部分,图片和标签
images, labels = batch

'''---------------------------------------------------------------------------------------------------------------------
 此部分内容为观察数据,查看数据的构成,理解数据'''
# print(len(train_set))  # 查看训练集大小  60000
# print(train_set.targets)  # 查看每个数据的标签,属于哪一个类  tensor([9, 0, 0,  ..., 3, 0, 5])
# print(train_set.targets.bincount())  # 查看每个标签的数量

sample = next(iter(train_set))  # iter返回一个数据流对象,next函数获取数据流中的下一个数据
# print(len(sample))  # 每个样本长度2,因为每个样本包含两部分 ,图片本身和对应标签
image, label = sample
# print(type(image))  # <class 'torch.Tensor'>  样本两个元素都是tensor类型
# print(type(label))  # <class 'int'>
# print(image.shape)  # torch.Size([1, 28, 28])  1颜色通道 28*28尺寸
# print(torch.tensor(label).shape)  # torch.Size([])  标量tensor

'''plt.imshow(image.squeeze(),cmap='gray')  展示图片
plt.show()'''

'''
# 查看数据经过dataloader加载后的样子
display_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10
)

batch = next(iter(display_loader))
loader_image, loader_label = batch
print(type(loader_image))  # <class 'torch.Tensor'>  转化成tensor形式
print(type(loader_label))  # <class 'torch.Tensor'>
print(
    loader_image.shape)  # torch.Size([10, 1, 28, 28])   符合前文所说,数据进入神经网络运算时有四个维度的输入  [batch_size,color_channel,height,width]'''

'''
grid = torchvision.utils.make_grid(loader_image,nrow=10)  展示图片
plt.figure(figsize=(15,15))
plt.imshow(grid.permute(1,2,0))
plt.show()'''


'''----------------------------------------------------------------------------------------此部分为构建CNN模型,面向对象OOP'''
class NetWork(nn.Module):  # 继承nn.Module类
    def __init__(self):
        super().__init__()  # 如果不super(),子类的init会覆盖父类的init
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # 上一层的输出数量就是下一层的输入数量
        # out_channels是输出通道的数量,也是滤波器的数量,一个滤波器对应一个输出    kernel_size是滤波器(filter)的高和宽,in_channel是滤波器的深度
        # 第一个in_channel = 1,取决于我们的训练集,就是颜色通道的数量,因为训练集是灰度图像,所以是1;如果是RGB图像,就是3
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)  # conv卷积层

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)  # linear线性层(密集层,全连接层)
        # 从卷积层到线性层,要将图片扁平化
        # in_channel=12*4*4,是上一层扁平化输出(flatten)的长度    12是因为上一个卷积层有12个输出,4*4是因为卷积和最大池化减少维度,图片尺寸有28*28变成了4*4
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)  # 最终10个输出取决于数据集,因为数据集有10种服装,10种类别
        '''一个经常出现的模式是，当我们增加额外的conv层时，我们的out_channels会增加，而当我们切换到线性层后，我们的out_features会缩小，因为我们会过滤掉输出类的数量。'''

    def forward(self, t):
        t = t  # 第一层是输入层,数据不进行转换,所以t=t

        t = self.conv1(t)  # 第二次卷积层
        t = F.relu(t)  # 激活函数
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # 最大池化,滤波器尺寸2,步长2

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        # t = F.softmax(t,dim=1)  不要再次调用softmax方法,因为损失函数已经隐式的调用了这个方法

        return t


network = NetWork()
# network.to(device)  # 模型放在GPU上
'''print('\n\n', network, '\n\n')  # 详细的打印了 继承的nn.module类提供的自定义字符串表示
print(network.conv1.weight)  # 访问权重参数,因为第一次,6个输出,也即6个滤波器,每个滤波器尺寸5*5,刚好6*5*5=120个参数(其实除了权重参数,还要偏置参数bias)'''


def num_of_correct(predict, labels):
    return predict.argmax(dim=1).eq(labels).sum().item()


total_correct = 0

'''------------------------------------------------开始训练 '''
optimizer = optim.Adam(network.parameters(), lr=0.01)  # 选用优化器,将参数传给优化器optimizer,其实就是将所有的权重参数传进去,后续可以用来更新
for epoch in range(5):
    for batch in train_loader:
        images, labels = batch
        # images = images.to(device)  # 数据放在GPU运行
        # labels = labels.to(device)
        predict = network(images)  # 预测,不用我们调用前向传播函数forward(),pytorch内置自动调用
        loss = F.cross_entropy(predict, labels)  # 计算损失函数
        total_correct += num_of_correct(predict, labels)  # 预测正确的个数
        optimizer.zero_grad()  # 将梯度设置为0,不然梯度会累加
        loss.backward()  # 计算梯度,反向传播
        optimizer.step()  # 更新
    print('loss:', loss.item())
    print('total_correct:', total_correct)
    print("正确率:", total_correct / len(train_set))
    total_correct = 0


'''----------------------------------------------对整个训练集进行预测'''
def get_all_prediction(model, loader):
    all_predict = torch.tensor([])
    for batch in loader:
        pictures, targets = batch
        # pictures = pictures.to(device)
        # targets = targets.to(device)
        # model.to(device)
        prediction = model(pictures)
        all_predict = torch.cat((all_predict, prediction))
    return all_predict

with torch.no_grad():  # 因为是最终预测,模型已经训练好,不需要梯度跟踪
    all_prediction = get_all_prediction(network, train_loader)  # torch.Size([60000, 10]) ,一个60000图片,每个图片对应10个预测值

# print(train_set.targets)
# print(all_prediction.argmax(dim=1))
stack = torch.stack((train_set.targets, all_prediction.argmax(dim=1)), dim=1)  # train_set.target是真实标签,all_prediction.argmax(dim=1)是预测标签的索引

# 创建一个混淆矩阵
cmt = torch.zeros(10, 10)
for p in stack:
    i,j = p.tolist()
    cmt[i][j] = cmt[i][j] + 1

print(cmt)
'''Confusion matrix, without normalization
[[5661    5   77   73    8    2  117    1   56    0]
 [  64 5774    5  128    5    1   20    0    3    0]
 [ 111    1 4692   82  768    1  299    0   46    0]
 [ 546   20   20 5216  138    0   56    0    4    0]
 [  21    6  364  297 4830    0  419    5   58    0]
 [  27    6    8    1    0 5665    2  213    8   70]
 [1871    9  612  127  498    0 2792    0   91    0]
 [   0    0    0    0    0   49    0 5846    3  102]
 [  40    1   23   20   13   15   25   15 5846    2]
 [   1    0    1    0    0   20    0  307    5 5666]]
'''


