import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import glob


class ListDataset(Dataset):
    """
    返回 图片路径、图片array、图片上的bbox（开辟10行存储，每行1个bbox，没有bbox的行值为0）
    img: (3, 160, 320)
    filled_labels: (10, 5)
    """
    def __init__(self, data_dir, mode='train'):
        img_dir = os.path.join(data_dir, "images", mode)
        self.img_paths = glob.glob(f'{img_dir}/*.*')
        self.lab_paths = [path.replace('images', 'labels').replace('png', 'txt').replace('jpg', 'txt') for path in self.img_paths]
        self.max_objects = 10 # 定义每一张图片最多含有的 box 数量

    def __getitem__(self, index):
        # 根据index获取对应的图片路径
        img_path = self.img_paths[index % len(self.img_paths)]
        img = cv2.imread(img_path) / 255.0 # 利用PIL Image读取图片, 然后转换成numpy数组
        img = np.transpose(img, (2, 0, 1)) # 将通道维度放置在首位(C,H,W)
        # 将numpy数组转换成tenosr, 数据类型为 float32
        img = torch.from_numpy(img).float()

        # 获取图片对应的 label 文件的路径
        lab_path = self.lab_paths[index % len(self.img_paths)]
        labels = np.loadtxt(lab_path).reshape(-1, 5)
        filled_labels = np.zeros((self.max_objects, 5)) # 创建50×5的占位空间
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        # 将 label 转化成 tensor
        filled_labels = torch.from_numpy(filled_labels)
        # 返回图片路径, 图片tensor, label tensor
        return img_path, img, filled_labels

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    batch_size = 2
    shuffle = True
    num_workers = 4
    dataset = ListDataset("E:\yolov3\\nlos-20231003", mode='train') #"E:\yolov3\nlos-20231003"  '/home/agent/Code/yolov3_pytorch/annotation/data_parallel'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    for img_paths, images, labels in dataloader:
        print(img_paths)
        print(images.size())
        print(labels.size())
        break
