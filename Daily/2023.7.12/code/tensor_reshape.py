import torch

t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4]
])
print(t.shape)
print(t.numel())  # 计算元素总数,我们在reshape张量的时候,需要考虑总元素的个数

t1 = t.reshape([1, 16])
t2 = t.reshape([8, 2])
print(t1, '\n', t2)
'''
不改变秩(rank)的reshape
tensor([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]]) 
 tensor([[1, 1],
        [1, 1],
        [2, 2],
        [2, 2],
        [3, 3],
        [3, 3],
        [4, 4],
        [4, 4]])'''

t3 = t.reshape([2, 2, 4])
t4 = t.reshape([1, 2, 8])
print(t3, '\n', t4)
'''
改变秩rank的reshape
tensor([[[1, 1, 1, 1],
         [2, 2, 2, 2]],

        [[3, 3, 3, 3],
         [4, 4, 4, 4]]]) 
 tensor([[[1, 1, 1, 1, 2, 2, 2, 2],
         [3, 3, 3, 3, 4, 4, 4, 4]]])'''

t5 = t.reshape(
    [1, 16]).squeeze()  # squeeze函数减少一个维度,但前提是,数据要以一条线的形式编排好,如果t5 = t.reshape([2,8]).squeeze(),两行八列不是一条线,无法减少维度
t6 = t.reshape([4, 4]).unsqueeze(dim=0)  # unsqueeze函数增加一个维度
print(t5, '\n', t6)
'''
tensor([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]) 
 tensor([[[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3],
         [4, 4, 4, 4]]])'''


def flatten(t):  # flatten函数是删除tensor的所有维度,剩下一维,   卷积神经网络用到flatten函数
    t = t.reshape([1,-1])  # 形状设为1行
    return t.squeeze()
t7 = flatten(t)
print(t7)
'''tensor([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])'''


# 合并tensor
tt1 = torch.tensor([
    [1,2],
    [3,4]
])
tt2 = torch.tensor([
    [5,6],
    [7,8]
])
tt3 = torch.cat((tt1,tt2), dim=0)  #dim=0按行合并,dim-1按列合并
tt4 = torch.cat((tt1,tt2),dim=1)
print(tt3,'\n',tt4)
'''
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]]) 
 tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])'''