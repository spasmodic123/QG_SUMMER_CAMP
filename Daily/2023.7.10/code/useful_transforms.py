from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import  SummaryWriter

# Totensor
writer = SummaryWriter('log_3')
image = Image.open("dataset/train/bees_image/36900412_92b81831ad.jpg")  #输出PIL类型
trans_tensor = transforms.ToTensor()
tensor_image = trans_tensor(image)
writer.add_image('Totensor',tensor_image)  # 转化为tensor类型

# 归一化
print("归一化前",tensor_image[0][0][0])
trans_normalize = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])  # 图片有三层,对应3个数,第一个列表均值,第二个列表标准差,不同的均值标准差会输出不同的结果
image_normalize = trans_normalize(tensor_image)  # 输入的是tensor类型
print("归一化后",image_normalize[0][0][0])
writer.add_image("image_normalize", image_normalize)


# resize,对图片进行缩放
print("缩放前",image.size)
trans_size = transforms.Resize((512, 512))  # 输入的是PIL而不是tensor类型图片,输入变化后的高和宽,如果只输入一个数字,那么就是等比缩放
image_resize = trans_size(image)
print("缩放后",image_resize.size)
image_resize = trans_tensor(image_resize)
writer.add_image("resize_image",image_resize)

# compose 将多个变化结合一起
trans_size_2 = transforms.Resize(700)
trans_compose = transforms.Compose([trans_size_2, trans_tensor])  # 多个变化结合压要求上一个变化的输出是下一个变化的输入,类型要匹配,否则报错
image_compose = trans_compose(image)
writer.add_image("compose_image", image_compose)

# randomcrop随机裁剪,在图片任意位置裁剪指定大小图片
trans_random = transforms.RandomCrop((90,50))
trans_ran_compose = transforms.Compose([trans_random,trans_tensor])
for i in range(6):
    image_randomcrop = trans_ran_compose(image)
    writer.add_image("image_randomcrop",image_randomcrop,i)


writer.close()


