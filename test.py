import numpy as np
import argparse
import tqdm
import time
import os
import torch
from torchsummary import summary
import cv2

from model import Darknet
from loss import YOLOLayer
from utils import set_seed, compute_mAP, non_max_suppression, get_detection_annotation
from dataset import ListDataset


"""
set params and log
set model
set dataset and dataloader

"""

# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="cpu if <0, or gpu id")
parser.add_argument("--seed", type=int, default=42, help="exp seed")
parser.add_argument("--model_path", type=str, default="output/20230928_105632/model/model-49.pth", help="path to model")
parser.add_argument("--data_dir", type=str, default="annotation/data_parallel", help="path to dataset")
parser.add_argument("--output_dir", type=str, default="output", help="path to results")
parser.add_argument("--batch_size", type=int, default=128, help="size of each image batch")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--img_size", type=int, nargs='+', default=[160, 320], help="size of each image dimension")
parser.add_argument("--save_freq", type=int, default=50, help="interval between saving model weights")
args = parser.parse_args()

device = f'cuda:{args.device}' if args.device >=0 and torch.cuda.is_available() else 'cpu'
set_seed(args.seed)
logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_dir = os.path.join(args.output_dir, logid)
res_dir = os.path.join(log_dir, 'result')
os.makedirs(res_dir, exist_ok=True)

# 创建模型
num_classes = 1
init_filter = 8
# model = Darknet(num_classes, init_filter).to(device)
# model.load_state_dict(torch.load(args.model_path, map_location=device))
model = torch.load(args.model_path)
summary(model, (3, 160, 320))

# 创建数据集
dataloader = torch.utils.data.DataLoader(
    ListDataset(args.data_dir, mode='val'), batch_size=args.batch_size
)
anchors = torch.Tensor([[[10, 13], [16, 30], [33, 23]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[116, 90], [156, 198], [373, 326]]])

# Loss
Loss1 = YOLOLayer(anchors[2], num_classes, iou_thres=args.iou_thres, device=device)
Loss2 = YOLOLayer(anchors[1], num_classes, iou_thres=args.iou_thres, device=device)
Loss3 = YOLOLayer(anchors[0], num_classes, iou_thres=args.iou_thres, device=device)

all_detections, all_annotations = [], []
img_path_res, detect_res, label_res = [], [], []
model.eval()
for img_path, imgs, targets in tqdm.tqdm(dataloader, desc="Detecting objects"):
    imgs = imgs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        y1, y2, y3 = model(imgs)
        pred1 = Loss1(y1)
        pred2 = Loss2(y2)
        pred3 = Loss3(y3)
        pred = torch.cat((pred1, pred2, pred3), dim=1)

    # 为了画图
    # img_path_res += img_path
    # detections = non_max_suppression(pred, num_classes, args.conf_thres, args.nms_thres)
    # detect_res += detections

    # 为了计算mAP
    batch_detections, batch_annotations = get_detection_annotation(pred, targets, args.conf_thres, args.nms_thres, num_classes, args.img_size)
    all_detections += batch_detections
    all_annotations += batch_annotations

# 以字典形式记录每一类的mAP值
average_precisions = compute_mAP(all_detections, all_annotations, num_classes, args.iou_thres)
print(f"mAP: {average_precisions[0]}")

H, W = args.img_size
color_pred = (0, 0, 255)  # 红色 (BGR颜色格式)
color_gt = (0, 255, 0)
box_thick = 2
for img_path, detections in zip(img_path_res, detect_res):
    img = cv2.imread(img_path)
    lab_path = img_path.replace('images', 'labels').replace('png', 'txt').replace('jpg', 'txt')
    labels = np.loadtxt(lab_path).reshape(-1, 5)
    lab_cls = labels[:, 0]
    labels = labels[:, 1:]
    annotation_boxes = np.zeros_like(labels)
    annotation_boxes[:, 0] = (labels[:, 0] - labels[:, 2] / 2) * W
    annotation_boxes[:, 1] = (labels[:, 1] - labels[:, 3] / 2) * H
    annotation_boxes[:, 2] = (labels[:, 0] + labels[:, 2] / 2) * W
    annotation_boxes[:, 3] = (labels[:, 1] + labels[:, 3] / 2) * H

    # 画GT
    for x1, y1, x2, y2 in annotation_boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_gt, box_thick)

    # 在图上画bbox和conf
    if detections is not None:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color_pred, box_thick)
    
    out_path = os.path.join(res_dir, img_path.split('/')[-1])
    cv2.imwrite(out_path, img)
