from torch.utils.data import Dataset
import os
from PIL import Image

image_path = "C:\\Python_Study\\deep_learning\\hymenoptera_data\\train\\ants"
dir_path = os.listdir(image_path)


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 根目录
        self.label_dir = label_dir  # 列表标签
        self.path = os.path.join(self.root_dir, self.label_dir)  # 列表地址
        self.images_path = os.listdir(self.path)  # 列表中所有图片编号,存到列表中

    def __getitem__(self, idx):
        image_name = self.images_path[idx]
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)  # 每一张图片的地址
        image = Image.open(image_item_path)  # 图片对象本身
        label = self.label_dir  # 图片标签
        return image, label

    def __len__(self):
        return len(self.images_path)

root_dir = "C:\\Python_Study\\deep_learning\\hymenoptera_data\\train"
label_dir = "ants"
bees_label_dir = "bees"
ant_dataset = MyData(root_dir,label_dir)  # 创建实例对象
bees_dataset = MyData(root_dir,bees_label_dir)

print(ant_dataset[0])
print(bees_dataset[0])

image1,label1 = ant_dataset[0]
image1.show()
image2,label2 = bees_dataset[1]
image2.show()

all_train_data = ant_dataset + bees_dataset