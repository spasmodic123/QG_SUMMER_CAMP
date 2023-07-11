from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import  SummaryWriter

#transform 类似一个工具箱,利用里面的工具,输入图片,输出我们想要的结果

image_path = "dataset/train/ants_image/0013035.jpg"
image = Image.open(image_path)
tensor_trans = transforms.ToTensor()  # 将PIL类型或numpy类型转化为tensor类型
tensor_image = tensor_trans(image)
print(tensor_image)

writer = SummaryWriter('logs_2')
writer.add_image("tensor_image", tensor_image)  # 上节课add_image是读取图片的numpy类型,这是读取tensor类型

writer.close()
