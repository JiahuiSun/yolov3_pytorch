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
    规定图片是png格式，tensor是npy格式
    """
    def __init__(self, data_dir, mode='train'):
        img_dir = os.path.join(data_dir, "images", mode)
        self.img_paths = glob.glob(f'{img_dir}/*.*')
        self.lab_paths = [path.replace('images', 'labels').replace('png', 'txt').replace('npy', 'txt') for path in self.img_paths]
        self.max_objects = 10  # 定义每一张图片最多含有的 box 数量
        self.img_fmt = True if self.img_paths[0][-3:] == 'png' else False

    def __getitem__(self, index):
        img_path = self.img_paths[index % len(self.img_paths)]
        if self.img_fmt:
            img = cv2.imread(img_path) / 255.0
        else:
            data = np.load(img_path)
        data = np.transpose(data, (2, 0, 1))  # 将通道维度放置在首位(C,H,W)
        data = torch.from_numpy(data).float()
        img, mask = data[0:1, ...], data[1:2, ...]

        lab_path = self.lab_paths[index % len(self.img_paths)]
        labels = np.loadtxt(lab_path).reshape(-1, 5)
        filled_labels = np.zeros((self.max_objects, 5))  # 创建50×5的占位空间
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        return img_path, img, mask, filled_labels

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    batch_size = 4
    shuffle = True
    num_workers = 4
    dataset = ListDataset('/home/agent/Code/ackermann_car_nav/data/radar_nlos_mat', mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    for img_paths, images, labels in dataloader:
        print(img_paths)
        print(images.size())
        print(labels.size())
        break
