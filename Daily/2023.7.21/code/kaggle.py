import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transform
from torchvision import datasets

import numpy as np
import pandas as pd

import os
import time
import shutil
import csv

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 给图片加上标签
label_path = '/kaggle/input/cassava-leaf-disease-classification/train.csv'
labels = pd.read_csv(label_path)

root_pth = '/kaggle/input/cassava-leaf-disease-classification'  # 根目录
data_dir = os.path.join(root_pth, 'train_images')  # 图片的目录
new_pth = '/kaggle/working/'  #
for root, dirs, files in os.walk(data_dir):
    for file in files:  # files就是一系列图片的名字,单个文件本身
        image_name = file

        label = labels[labels['image_id'] == image_name][
            'label'].values.item()  # int type  执行一个判断语句['iamge_id==image_name'],寻找与图片匹配的label,在['label']返回label的值

        out_dir = os.path.join(new_pth, str(label))  # 不同的标签存放在不同的文件内
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        to_path = os.path.join(out_dir, file)  # 新保存的文件目录
        from_path = os.path.join(data_dir, file)  # 原来图片的目录
        shutil.copy(from_path, to_path)  # shutil.move 将加上label后的照片放在新的目录


class MyData(Dataset):  # pytorch读取图片的方法
    def __init__(self,root,label):
        self.root = root  # 所有图片的跟目录
        self.label = label  # 图片的标签目录,图片有多个标签
        self.transform = transform.Compose([
            transform.Resize((256,256)),
            transform.ToTensor(),
            transform.Normalize(mean=[0,0,0],std=[1,1,1])
        ])
        self.path = os.path.join(self.root,str(self.label))
        self.images_names = os.listdir(self.path)  # 一个列表,储存所以图片的名字
    def __getitem__(self,index):
        image_name = self.images_names[index]
        image_path = os.path.join(self.root,str(self.label),image_name)  # 获取图片的路径
        image = Image.open(image_path)  # 图片本身
        image = self.transform(image)  # 需要将PIL格式转化为tensor格式
        label = self.label  # 图片对应的标签
        return image,label
    def __len__(self):
        return len(self.images_names)  # 返回数据集大小


train_set1 = MyData("/kaggle/working/",1)
train_set2 = MyData("/kaggle/working/",2)
train_set3 = MyData("/kaggle/working/",3)
train_set4 = MyData("/kaggle/working/",4)
train_set0 = MyData("/kaggle/working/",0)  #
train_set = train_set1 + train_set2 + train_set3 + train_set4 + train_set0
print(len(train_set))


# 搭建网络框架
class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=60, kernel_size=5)
        self.fc1 = nn.Linear(in_features=60 * 61 * 61, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=5)

    def forward(self, t):
        t = t

        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 60 * 61 * 61)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        return t


def get_correct_num(predictions,labels):
    return predictions.argmax(dim=1).eq(labels).sum().item()


# 搭建网络并且训练
total_correct = 0
network = NetWork()
network.to(device)  # GPU
optimizer = optim.Adam(network.parameters(), lr=0.01)
for epoch in range(5):
    for batch in train_loader:
        images, labels = batch
        images = images.to('cuda')  # GPU
        labels = labels.to('cuda')
        predictions = network(images)
        total_correct += get_correct_num(predictions, labels)
        loss = F.cross_entropy(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('loss', loss.item())
    print('accuracy', total_correct / len(train_set))
    total_correct = 0


class TEST(Dataset):  # 定义读取测试集的Dataset类
    def __init__(self, root):
        self.root = root  # 根目录
        self.transforms = transform.Compose([
            transform.Resize((256, 256)),
            transform.ToTensor(),
            transform.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        ])
        self.images_names = os.listdir(root)  # 将所有图片的名字放在列表中

    def __getitem__(self, index):
        one_image_name = self.images_names[index]
        one_image_path = os.path.join(self.root, one_image_name)
        image = Image.open(one_image_path)
        image = self.transforms(image)
        label = 1
        return image, label

    def __len__(self):
        return len(self.images_names)


root = '/kaggle/input/cassava-leaf-disease-classification/test_images'
test_set = TEST(root)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=200, shuffle=False)


# 预测
# 最终预测函数
def get_all_prediction(model, loader):
    all_predict = torch.tensor([]).cuda()
    all_prediet = all_predict.to(device)
    for batch in loader:
        pictures, targets = batch
        pictures = pictures.to(device)
        targets = targets.to(device)
        model = model.to(device)
        prediction = model(pictures)
        max_index = prediction.argmax(dim=1)  # 求出最大值的索引,也就是预测结果
    return max_index


with torch.no_grad():  # 因为是最终预测,模型已经训练好,不需要梯度跟踪
    all_prediction = get_all_prediction(network, test_loader)

print(all_prediction.item())

# 写入文件
f = open('sample_submisson.csv', 'w', encoding='utf-8')
test_images_names = os.listdir('/kaggle/input/cassava-leaf-disease-classification/test_images')
csv_writer = csv.writer(f)
csv_writer.writerow(['image_id', 'label'])
for i in range(len(test_images_names)):
    csv_writer.writerow([test_images_names[i], all_prediction[i].item()])