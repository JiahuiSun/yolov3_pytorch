import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np


class CustomDataset(Dataset):
    """
    把数据交给我，我帮你load
    1. 把训练数据集路径给我，我一一加载图片
    2. 再一一加载bbox
    """
    def __init__(self, data_path):
        with open(data_path) as f:
            img_labels = f.readlines()
        self.img_labels = [img_lab.strip() for img_lab in img_labels]

    def bbox_iou_data(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]
        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors):
        # TODO: 为什么一定是矩形呢？
        label = [np.zeros((train_output_sizes[i], train_output_sizes[i], 3,
                        5 + num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))
        for bbox in bboxes:
            bbox_coor = bbox[:4]  # xyxy
            bbox_class_ind = bbox[4]  # cls
            onehot = np.zeros(num_classes, dtype=np.float32)  # one-hot
            onehot[bbox_class_ind] = 1.0
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)  # xywh
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
            iou = []
            for i in range(3):
                anchors_xywh = np.zeros((3, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = anchors[i]
                iou_scale = self.bbox_iou_data(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / 3)
            best_anchor = int(best_anchor_ind % 3)
            xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
            # 防止越界
            grid_r = label[best_detect].shape[0]
            grid_c = label[best_detect].shape[1]
            xind = max(0, xind)
            yind = max(0, yind)
            xind = min(xind, grid_r - 1)
            yind = min(yind, grid_c - 1)
            label[best_detect][yind, xind, best_anchor, :] = 0
            label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
            label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
            label[best_detect][yind, xind, best_anchor, 5:] = onehot
            bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
            bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
            bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def image_preporcess(self, image, target_size, gt_boxes=None):
        # 这里改变了一部分原作者的代码。可以发现，传入训练的图片是bgr格式
        ih, iw = target_size
        print("target shape: ",ih,iw)
        h, w = image.shape[:2]
        print("image shape: ",h,w)
        M, h_out, w_out = training_transform(h, w, ih, iw)
        # 填充黑边缩放
        letterbox = cv2.warpAffine(image, M, (w_out, h_out))
        pimage = np.float32(letterbox) / 255.
        if gt_boxes is None:
            return pimage
        else:
            scale = min(iw / w, ih / h)
            nw, nh = int(scale * w), int(scale * h)
            dw, dh = (iw - nw) // 2, (ih - nh) // 2
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return pimage, gt_boxes

    def parse_annotation(self, img_path):
        """输入一张图片地址，返回该图片和它上的bbox
        """
        line = img_path.split()
        img_path = "./annotation/data_parallel/images/" + line[0]
        if not os.path.exists(img_path):
            raise KeyError("%s does not exist ... " % img_path)
        img = cv2.imread(img_path)

        # 没有标注物品，即每个格子都当作背景处理
        # TODO: 这个变量是什么意思？没有标注物体，确实存在这种现象，那么这个bbox还有用吗？
        exist_boxes = True
        if len(line) == 1:
            bboxes = np.array([[10, 10, 101, 103, 0]])
            exist_boxes = False
        else:
            bboxes = np.array([int(float(x)) for x in line[1:]])
            bboxes = bboxes.transpose(1, 0)
        return img, bboxes, exist_boxes

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        """
        给我1张图片的路径，我把图片读进来，label解析出来
        # TODO: 就算不提前加载所有图片，至少可以提前把每张图片的label给解析出来
        # 可以一开始就把label给解析出来，其实就是Nx3x8x8x256，把这个保存到磁盘里，就像图片一样
        # 第二次就不需要再重新解析，直接读取即可
        """
        img_path_lab = self.img_labels[index]
        # 一一加载图片，并解析label
        image, bboxes, exist_boxes = self.parse_annotation(img_path_lab)
        label_sbbox, label_mbbox, label_lbbox, sbbox, mbbox, lbbox = self.preprocess_true_boxes(bboxes, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors)
        sample = {
            'img': image,
            'label_sbbox': label_sbbox,
            'label_mbbox': label_mbbox,
            'label_lbbox': label_lbbox,
            'sbbox': sbbox,
            'mbbox': mbbox,
            'lbbox': lbbox
        }
        return sample


batch_size = 64
shuffle = True
num_workers = 4

data = torch.zeros((128, 3, 24, 24))
labels = torch.zeros((128, 10))
dataset = CustomDataset(data, labels)  # 自定义的数据集
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

for batch in dataloader:
    inputs = batch['data']
    targets = batch['label']

    # 在这里执行模型的前向传播和训练步骤
    print(inputs.size())
    print(targets.size())
