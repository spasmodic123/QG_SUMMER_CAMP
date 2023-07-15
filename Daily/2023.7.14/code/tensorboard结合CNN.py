import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


# 获取正确的预测的数量
def get_number_correct(prdiction, true_label):
    return prdiction.argmax(dim=1).eq(true_label).sum().item()


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


# 导入数据并加载
train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
# 而是不同的batch,不同的学习率造成的影响
batch_size_list = [100, 1000, 10000]
lr_list = [0.01, 0.001, 0.0001, ]

for batch_size in batch_size_list:
    for lr in lr_list:
        network = Network()  # 创建模型对象
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)  # 加载数据
        batch = next(iter(train_loader))  # 获取数据流中的一组数据
        images, true_labels = batch  # 解压数据,分为图片本身和labels
        optimizer = optim.Adam(lr=lr, params=network.parameters())  # 确定学习率,以及需要更新的参数

        # 创建能够在tensorboard中显示的图像网格,因为tensorboard的add_image需要特定的图片形式(torch.Tensor, numpy.ndarray, or string/blob-name_
        grid = torchvision.utils.make_grid(images)
        tb = SummaryWriter(f'batch_size={batch_size}  lr={lr}')  # 命名
        tb.add_image('images', grid)  # 将第一批图像放在grid中进行展示
        tb.add_graph(network, images)  # 在tensorboard中看见network结构的可视化图

        # 训练模型
        for epoch in range(5):
            total_correct = 0
            for batch in train_loader:
                prediction = network(images)
                loss = F.cross_entropy(prediction, true_labels)  # 计算损失函数
                optimizer.zero_grad()  # 清零梯度,否则会累加
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新权重

                total_correct += get_number_correct(prediction, true_labels)  # 将每一个batch的正确预测数量加起来得到一个epoch的正确预测数量

            tb.add_scalar('total_correct', total_correct)  # 为图像增加数值
            tb.add_scalar('accuracy', total_correct / len(train_set))

            for name, weight in network.named_parameters():
                tb.add_histogram(name, weight, epoch)  # 参数对应名字,数值,步数,  绘制直方图,查看每走一个步数,数值的变化情况,也即是不同的周期,权重的变化
                tb.add_histogram(f'{name}.grad', weight.grad, epoch)

            print("epoch:", epoch, "total_correct:", total_correct)

        tb.close()