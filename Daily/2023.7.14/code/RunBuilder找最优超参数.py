# RunBuilder类的编写允许我们使用不同的参数值生成多个运行
from itertools import product
from collections import OrderedDict
from collections import namedtuple
import time
import pandas as pd

import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms

parameters = dict(  # 键值对
    lr=[.01, .001],
    batch_size=[10, 100, 1000],
    shuffle=[True, False]
)
param_values = [v for v in parameters.values()]
# print(param_values)
'''[[0.01, 0.001], [10, 100, 1000], [True, False]]'''

# product产生笛卡尔积, For example, product(A, B) returns the same as:  ((x,y) for x in A for y in B).
for lr, batch, shuffle in product(*param_values):
    # print(lr,batch,shuffle)
    '''0.01 10 True
0.01 10 False
0.01 100 True
0.01 100 False
0.01 1000 True
0.01 1000 False
0.001 10 True
0.001 10 False
0.001 100 True
0.001 100 False
0.001 1000 True
0.001 1000 False
有这个操作,同样在训练神经网络的时候测试不同的参数就可以减少使用循环的数量-----------------------------------------------------------'''


# RunBuilder 类的编写
class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())  # namedtuple返回带有命名字段的新元组子类, Run就是名字
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


params = OrderedDict(  # 确定参数
    lr=[0.01, 0.001],
    batch_size=[1000, 10000]
)

runs = RunBuilder.get_runs(params)
# print(runs)   [Run(lr=0.01, batch_size=1000), Run(lr=0.01, batch_size=10000), Run(lr=0.001, batch_size=1000), Run(lr=0.001, batch_size=10000)]

for run in runs:
    # print(run,run.lr,run.batch_size)
    pass
# Run(lr=0.01, batch_size=1000) 0.01 1000
# Run(lr=0.01, batch_size=10000) 0.01 10000
# Run(lr=0.001, batch_size=1000) 0.001 1000
# Run(lr=0.001, batch_size=10000) 0.001 10000

# 创建RunBuilder类之后,comment可以表示为:(comment是传递给SummaryWriter的名字参数)
for run in RunBuilder.get_runs(params):
    comment = f'-{run}'
    # print(comment)


# -Run(lr=0.01, batch_size=1000)
# -Run(lr=0.01, batch_size=10000)
# -Run(lr=0.001, batch_size=1000)
# -Run(lr=0.001, batch_size=10000)


# 构建RunManager参数可以实验对于大量超参数的实验
class RunManager():
    def __init__(self):
        self.epoch_count = 0  # 周期数
        self.epoch_loss = 0  # 每个周期的loss
        self.epoch_duration = 0  # 每个周期的运算时间,运行时间长短也是我们衡量超参数的标准之一
        self.epoch_start_time = None  # 周期开始时间
        self.epoch_correct_number = None  # 每个周期正确的预测数量,计算精确率accuracy用到,衡量超参数的指标

        self.run_params = None  # 每一个run包含一组测试的超参数   run的形式为 Run(lr=0.01, batch_size=1000)
        self.run_count = 0  # 测试的超参数组别
        self.run_data = []  # 存放最终的测试数据
        self.run_start_time = None

        self.network = None  # 神经网络模型
        self.loader = None  # 加载器,转化成数据为可以输入network进行运算的类型
        self.tb = None  # 将数据导入tensorboard进行可视化

    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()

        self.run_params = run  # 第几组的测试,就传递第几组的测试参数,传递的类型  Run(lr=0.01, batch_size=1000)
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')  # comment是名字,长这样Run(lr=0.01, batch_size=1000)

        images, labels = next(iter(self.loader))  # 获取加载器的数据
        grid = torchvision.utils.make_grid(images)  # 转化为可以输入到tensorboard的类型

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0  # 周期数归零,下一组数据从第一个周期开始

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0  # loss归零,再次计算的是本周期的loss
        self.epoch_correct_number = 0  # 正确预测数量也归零,再次计算本周期的正确预测数量

    def end_epoch(self):
        self.epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / (len(self.loader.dataset) / int(self.run_params.batch_size))
        # 损失函数除以训练集大小因为,每一个把batch都会计算loss,一个epoch有多个batch,一个epoch的loss是多个batch的loss的平均值
        accuracy = self.epoch_correct_number / len(self.loader)

        self.tb.add_scalar('loss', loss, self.epoch_count)
        self.tb.add_scalar('accuracy', accuracy, self.epoch_count)

        # 绘制直方图,查看不同权重的分布范围
        for name, params in self.network.named_parameters():
            self.tb.add_histogram(name, params, self.epoch_count)

        results = OrderedDict()  # 返回一个字典,只不过是可以记忆插入元素的顺序的字典
        results['run'] = self.run_count  # 每个周期结束,统一保存数据
        results['epoch'] = self.epoch_count
        results['loss'] = self.epoch_loss
        results['accuracy'] = accuracy
        results['epoch_duration'] = self.epoch_duration
        results['run_duration'] = run_duration

        for k, v in self.run_params._asdict().items():  # 因为run_params的形式是Run(lr=0.01, batch_size=1000),_asdict转化为字典
            results[k] = v  # 保存本次测试的超参数

        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data)

    def track_loss(self, loss):  # 多个batch的loss加起来,后面再求平均就是一个epoch的loss
        self.epoch_loss += loss.item()

    def get_num_correct(self,predictions, labels):
        return predictions.argmax(dim=1).eq(labels).sum().item()

    def track_correct_numbers(self, predictions, labels):
        self.epoch_correct_number += self.get_num_correct(predictions, labels)

    # 将数据保存到文件中
    def save(self, filename):
        pd.DataFrame.from_dict(self.run_data).to_csv(f'{filename}.csv')  # f加''加{}代表可以解析的任何数据类型,f'{x}'会解析成x本身


# 定义network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = t
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12 * 4 * 4)  # t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))

        t = F.relu(self.fc2(t))

        t = self.out(t)
        return t

# 使用RunManager类和RunBuilder类可以使得程序乘以拓展
params = OrderedDict(
    lr=[0.01, 0.001],
    batch_size=[1000, 2000]
)

manager = RunManager()

# 加载数据
train_set = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

for run in RunBuilder.get_runs(params):
    network = Network()
    loader = torch.utils.data.DataLoader(train_set,batch_size=run.batch_size)  # 转化格式后的数据
    optimizer = optim.Adam(network.parameters(),lr=run.lr)

    manager.begin_run(run, network, loader)  # 开始测试,传递多组不同的参数组合,试图找出最优的超参数
    for epoch in range(5):
        manager.begin_epoch()  # 周期开始
        for batch in loader:
            images,labels = batch
            predictions = network(images)
            loss = F.cross_entropy(predictions,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            manager.track_loss(loss) # 将每一个batch的loss加起来,到了end_epoch取平均值就是一个epoch的loss
            manager.track_correct_numbers(predictions, labels)

        manager.end_epoch()  # 结束一个周期

    manager.end_run()  # 全部参数组合测试完成

manager.save('Run_results')  # 将测试结果保存到文件中
