import torch
import numpy as np

t = torch.Tensor()
print(t.type(),t.dtype,t.device,t.layout)  # torch.FloatTensor  torch.float32  cpu  torch.strided

data = np.array([1,2,3])
o1 = torch.Tensor(data)
o2 = torch.tensor(data)
o3 = torch.as_tensor(data)
o4 = torch.from_numpy(data)

# data[0] = 0   加了这个操作,下面输出变成了
'''tensor([1., 2., 3.]) 
 tensor([1, 2, 3], dtype=torch.int32) 
 tensor([0, 2, 3], dtype=torch.int32)   因为as_tensor()和from)numpy() 会实时共享data的数据,而Tensor()和tensor()只是复制了data的数据
 tensor([0, 2, 3], dtype=torch.int32)'''

print(o1,'\n',o2,'\n',o3,'\n',o4)
'''tensor([1., 2., 3.])  浮点型
 tensor([1, 2, 3], dtype=torch.int32) 整型
 tensor([1, 2, 3], dtype=torch.int32) 
 tensor([1, 2, 3], dtype=torch.int32)'''



o5 = torch.eye(2)  # 创建一个2D tensor,是个对角矩阵
print(o5)
'''tensor([[1., 0.],
        [0., 1.]])'''

o6 = torch.zeros([2,3])  # 创建指定大小的 0 tensor
print(o6)
'''tensor([[0., 0., 0.],
        [0., 0., 0.]])'''

o7 = torch.ones([2,2])  # 创建指定大小的 1 tensor
print(o7)
'''tensor([[1., 1.],
        [1., 1.]])'''

o8 = torch.rand([2,2])  # 创建指定大小tensor,分量值任意
print(o8)
'''tensor([[0.1904, 0.8529],
        [0.8422, 0.3881]])'''