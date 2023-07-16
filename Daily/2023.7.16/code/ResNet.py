import torch.nn as nn
import torch


class BasicBlock(nn.Module):  # 18层和34层
    expansion = 1  # 残差结构第三层卷积核个数是第一层的1倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        # downsample代表下采样结构,就是我们残差结构的残差分支(也就是虚线部分),起到升维的作用
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU

        self.conv2 = nn.Conv2d(in_channel=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self,x):
        identity = x  # identity为特征矩阵
        if self.downsample is not None:
            indentity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out += indentity  # 最后将主分支和残差分支的特征矩阵相加
        out = self.relu(out)  # 残差结构最后一次先将特征矩阵相加,然后再是要激活函数

        return out


class Bottleneck(nn.Module):  # 50层,101层,152层
    expansion = 4  # 残差结构的第三层卷积核个数是第一层的4倍

    def __init__(self, in_channel, out_channle, stride=1,downsamplle=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel=in_channel,out_channels=out_channle,kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channle)

        self.conv2 = nn.Conv2d(in_channel=out_channle,out_channels=out_channle, kernel_size=3,padding=1,stride=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channle)

        # 残差结够最后一层的卷积核个数是第一层的4倍,所以输出个数也是第一层的4倍    但无论怎样最终输出一个特征矩阵,包含了一个batch多张图片的信息
        self.conv3 = nn.Conv2d(in_channel=out_channle,out_channels=out_channle*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channle*self.expansion)

        self.relu = nn.ReLU
        self.downsample = downsamplle

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out += identity
            out = self.relu(out)

            return out


class ResNet(nn.Module):
    def __init__(self,block, block_num, num_classes=1000,includ_top=True):
        # block是根据我们的层数选用不同的残差结构,class_num是一个列表,返回的是不同层(conv2_x,conv3_x,conv4_x)使用的残差结构的数目
        super(ResNet, self).__init__()
        self.include_top = includ_top
        self.in_channel = 64  # 残差结构的输入通道为64,不是刚开始加载图片的通道,刚开始的RGB图像的通道为3

        self.conv1 = nn.Conv2d(3,out_channels=self.in_channel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self.make_layer(block,64,block_num[0])  # 对应笔记的conv2-x  64代表第一层输出64通道
        self.layer2 = self.make_layer(block, 128, block_num[1], stride=2)  # 对应笔记的conv3-x 第一层输出128通道
        self.layer3 = self.make_layer(block, 256, block_num[2], stride=2)  # 对应笔记的conv4-x  第一层输出256通道
        self.layer4 = self.make_layer(block, 512, block_num[3], stride=2)  # 对应笔记的conv5-x  第一层输出512通道

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d((1,1))  # 平均池化
            self.fc = nn.Linear(512*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal(m.weight,mode='fan_out',nonlinearity=True)
    def _make_layer(self,block,channel,block_num,stride=1):
        # block根据层数不同,选择少层数的残差结构或多层数的残差结构
        # channel是conv2_x或conv3_x或conv4_x或conv5_x的第一层卷积核的个数,也即是输出通道个数
        # block_num就是一列表,对应conv2_x或conv3_x或conv4_x或conv5_x各自的层数
        downsample = None
        if stride !=1 or self.in_channel != channel * block.expansion:  # 18层和34层跳过该语句,50,101层进入该语句,进行下采样
            # self.in_channel固定了64,而18和34层对应expansion=1,onv2_x或conv3_x或conv4_x或conv5_x第一层和最后一层卷积核个数相等,也即输出通道相等
            # 而50和101层,对应expansion=1,conv2_x或conv3_x或conv4_x或conv5_x,最后一层的卷积核个数是第一层的4倍
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,channel*block.expansion,kernel_size=1,stride=stride,bias=False),
                # 下采样只升维或降维,改变深度,不改变高和宽,所以kernel_size=1,stride=stride=1
                nn.BatchNorm2d(channel*block.expansion)
            )

        layers = []
        layers.append((block(self.in_channel,channel,downsample=downsample,stride=stride)))
        self.in_channel = channel * block.expansion  # 18层和34层不会改变,50和101层,  conv2_x或conv3_x或conv4_x或conv5_x,最后一层的卷积核个数是第一层的4倍

        for _ in range(1,block_num):  # 构建实线残差结构部分
            layers.append(block(self.in_channel))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)  # 第一层卷积层
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 进入残差结构  分别对应conv2_x或conv3_x或conv4_x或conv5_x,
        x =self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x,1)  # 传递给线性层之前进行扁平化操作
            x = self.fc(x)


def resent34(num_classes=1000,include_top=True):  # 34层ResNet
    return ResNet(BasicBlock,[3,4,6,3],num_classes=num_classes,include_top=include_top)

def resent101(num_classes=1000,include_top=True):  # 101层ResNet
    return ResNet(Bottleneck,[3,4,23,3],num_classes=num_classes,include_top=include_top)