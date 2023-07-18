## 利用LSTM ,通过sin函数预测cos函数
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, ):
        # hidden_size代表LSTM层输出的维度,也即是全连接层接收的输入维度 , num_layers为隐藏层的层数
        super(LSTM, self).__init__()

        self.letm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.letm(_x)  # 接收LSTM层的输出
        seq_size, batch_size, hidden_size = x.shape
        # 进入全连接层要扁平化操作
        # hidden_size代表LSTM层输出的维度,也即是全连接层接收的输入维度
        x = x.view(seq_size * batch_size, hidden_size)  # view函数相当于reshape函数
        x = self.fc1(x)
        x = x.view(seq_size, batch_size, -1)

        return x


if __name__ == '__main__':
    data_len = 200
    t = np.linspace(0, 12 * np.pi, data_len)  # 时间序列
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    dataset = np.zeros((data_len, 2))
    dataset[:, 0] = sin_t
    dataset[:, 1] = cos_t
    dataset = dataset.astype('float32')

    # 划分训练集和测试集
    data_ratio = 0.8
    train_x = dataset[:int(data_ratio * data_len), 0]
    train_y = dataset[:int(data_ratio * data_len), 1]
    train_time = t[:int(data_ratio * data_len)]

    test_x = dataset[int(data_ratio * data_len):, 0]
    test_y = dataset[int(data_ratio * data_len):, 1]
    test_time = t[int(data_ratio * data_len):]

    # 改变数据的形状,才能输入网络  x:[seq_length, batch_size, input_size]
    input_feature = 1
    output_feature = 1
    train_x = train_x.reshape(-1, 5, input_feature)
    train_y = train_y.reshape(-1, 5, output_feature)
    test_x = test_x.reshape(-1, 5, input_feature)
    test_y = test_y.reshape(-1, 5, output_feature)

    # 改变数据为tensor格式,才能输入网络
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    # test_y = torch.tensor((test_y))  test_y最终验证,不用转化形式
    print(train_y.type)

    model = LSTM(input_size=input_feature, hidden_size=16, num_layers=1, output_size=output_feature)  # 面向对象,创建模型
    loss_function = nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器

    epochs_all = 10000  # epoch很大因为是一直训练,达到预期效果才停下来
    for epoch in range(epochs_all):
        output = model(train_x)
        loss = loss_function(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < 1e-4:  # 0.0001
            print('epoch_num=[{}/{}],  loss={}'.format(epoch + 1, epochs_all, loss.item()))
            print('成功达到预期')
            break
        elif (epoch + 1) % 100 == 0:  # 每100个周期打印一次,观察情况
            print('epoch_num=[{}/{}],  loss={}'.format(epoch + 1, epochs_all, loss.item()))

    # 开始测试训练成果
    # 训练集的训练成果
    train_prediction = model(train_x)
    train_prediction = train_prediction.view(-1, output_feature).data.numpy()
    # 测试集的训练成果
    test_prediction = model(test_x)
    test_prediction = test_prediction.view(-1, output_feature).data.numpy()

    # 画图查看
    # 给原始的训练集画图
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    axs[0][0].plot(t[0:int(data_ratio * data_len)], dataset[0:int(data_ratio * data_len), 0], label='sin')
    axs[0][0].plot(t[0:int(data_ratio * data_len)], dataset[0:int(data_ratio * data_len), 1], label='cos')
    axs[0][0].plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5')
    axs[0][0].plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t=6.8')
    # 不同的时间,sin的值相同,cos却不同,说明我们不仅要考虑值,还要考虑时间序列
    # 这也是LSTM与其他网络不同的地方

    # 训练集成果
    axs[0][1].plot(t[int(data_ratio * data_len):], dataset[int(data_ratio * data_len):, 0])
    axs[0][1].plot(t[int(data_ratio * data_len):], test_prediction)

    plt.show()
