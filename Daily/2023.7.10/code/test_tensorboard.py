from torch.utils.tensorboard import  SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter('logs')  # 创建新的文件夹,存放tensorboard事件的文件,  打开文件方法,终端输入tensorboard --logdir=文件名 --port=端口

'''for i in range(100):
    writer.add_scalar("y=2x function", 2*i, i)  参数:命名,值,步数
writer.close()'''

image_pil = Image.open("hymenoptera_data/train/ants/0013035.jpg")
print(type(image_pil))  # <class 'PIL.JpegImagePlugin.JpegImageFile'> PIL不符合add_image函数的图像类型
image_np = np.array(image_pil)  #转化为numpy类型的图片
print(type(image_np))
print(image_np.shape)  # 3通道在最后一位,需要dataformats='HWC',  从PIL转化到numpy,需要指定shape中每一个数字(维度)代表的含义
writer.add_image("test", image_np, 2, dataformats='HWC')  #参数:命名,图片对象,步数,dataformats

writer.close()