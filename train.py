import os
import argparse
import torch
import time
from torchsummary import summary
from tqdm import tqdm
import wandb
import numpy as np
import pickle

from model import Darknet
from dataset import ListDataset
from loss import YOLOLayer
from utils import set_seed, get_detection_annotation, compute_mAP, non_max_suppression


# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="cpu if <0, or gpu id")
parser.add_argument("--seed", type=int, default=42, help="exp seed")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--data_dir", type=str, default="annotation/data_parallel", help="path to dataset")
parser.add_argument("--output_dir", type=str, default="output", help="path to results")
parser.add_argument("--batch_size", type=int, default=128, help="size of each image batch")
parser.add_argument("--iou_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
parser.add_argument("--img_size", type=int, nargs='+', default=[160, 320], help="size of each image dimension")
parser.add_argument("--save_freq", type=int, default=50, help="interval between saving model weights")
args = parser.parse_args()

device = f'cuda:{args.device}' if args.device >=0 and torch.cuda.is_available() else 'cpu'
set_seed(args.seed)
logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_dir = os.path.join(args.output_dir, logid)
model_dir = os.path.join(log_dir, 'model')
res_dir = os.path.join(log_dir, 'result')
os.makedirs(res_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
wandb.init(
    project=f"YOLOv3",
    name=f"{logid}",
    config=vars(args)
)

# 创建模型
num_classes = 1
init_filter = 8
model = Darknet(num_classes, init_filter).to(device)
summary(model, (3, 160, 320))

# 创建数据集
train_dataloader = torch.utils.data.DataLoader(
    ListDataset(args.data_dir, mode='train'), batch_size=args.batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    ListDataset(args.data_dir, mode='val'), batch_size=args.batch_size
)
anchors = torch.Tensor([[[10, 13], [16, 30], [33, 23]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[116, 90], [156, 198], [373, 326]]])

# loss function and optimizer
Loss1 = YOLOLayer(anchors[2], num_classes, iou_thres=args.iou_thres, device=device)
Loss2 = YOLOLayer(anchors[1], num_classes, iou_thres=args.iou_thres, device=device)
Loss3 = YOLOLayer(anchors[0], num_classes, iou_thres=args.iou_thres, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

"""
训练过程中，保存最后1个epoch的验证集上的输出；
再加载一下验证集，保存输出；
"""

for epoch in tqdm(range(args.epochs)):
    loss1_list = {
        'loss_x': [],
        'loss_y': [],
        'loss_w': [],
        'loss_h': [],
        'loss_conf': [],
        'loss_cls': []
    }
    loss2_list = {
        'loss_x': [],
        'loss_y': [],
        'loss_w': [],
        'loss_h': [],
        'loss_conf': [],
        'loss_cls': []
    }
    loss3_list = {
        'loss_x': [],
        'loss_y': [],
        'loss_w': [],
        'loss_h': [],
        'loss_conf': [],
        'loss_cls': []
    }

    train_annotations, train_detections = [], []
    model.train()
    for img_path, imgs, targets in train_dataloader:
        # imgs: [B, 3, 416, 416]
        # targets: [B, 50, 5]
        optimizer.zero_grad()
        imgs = imgs.to(device)
        targets = targets.to(device)
        y1, y2, y3 = model(imgs)
        loss1_dict, pred_bbox1 = Loss1(y1, targets)
        loss2_dict, pred_bbox2 = Loss2(y2, targets)
        loss3_dict, pred_bbox3 = Loss3(y3, targets)
        loss = loss1_dict[0] + loss2_dict[0] + loss3_dict[0]
        loss.backward()
        optimizer.step()
        # NOTE: 计算mAP
        # mAP一直是0，为了判断是不是计算mAP的函数有误，可以打印一下预测的结果，与ground truth，然后手动看看，到底是不是0
        pred = torch.cat((pred_bbox1.data, pred_bbox2.data, pred_bbox3.data), dim=1)
        batch_detections, batch_annotations = get_detection_annotation(pred, targets, args.conf_thres, args.nms_thres, num_classes, args.img_size)
        train_detections += batch_detections
        train_annotations += batch_annotations

        loss1_list['loss_x'].append(loss1_dict[1])
        loss1_list['loss_y'].append(loss1_dict[2])
        loss1_list['loss_w'].append(loss1_dict[3])
        loss1_list['loss_h'].append(loss1_dict[4])
        loss1_list['loss_conf'].append(loss1_dict[5])
        loss1_list['loss_cls'].append(loss1_dict[6])

        loss2_list['loss_x'].append(loss2_dict[1])
        loss2_list['loss_y'].append(loss2_dict[2])
        loss2_list['loss_w'].append(loss2_dict[3])
        loss2_list['loss_h'].append(loss2_dict[4])
        loss2_list['loss_conf'].append(loss2_dict[5])
        loss2_list['loss_cls'].append(loss2_dict[6])

        loss3_list['loss_x'].append(loss3_dict[1])
        loss3_list['loss_y'].append(loss3_dict[2])
        loss3_list['loss_w'].append(loss3_dict[3])
        loss3_list['loss_h'].append(loss3_dict[4])
        loss3_list['loss_conf'].append(loss3_dict[5])
        loss3_list['loss_cls'].append(loss3_dict[6])
    # 计算训练集的指标
    train_average_precisions = compute_mAP(train_detections, train_annotations, num_classes, args.iou_thres)

    # 测试一下验证集的指标
    model.eval()
    val_annotations, val_detections = [], []
    for img_path, imgs, targets in val_dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            y1, y2, y3 = model(imgs)
            pred_bbox1 = Loss1(y1)
            pred_bbox2 = Loss2(y2)
            pred_bbox3 = Loss3(y3)
            pred = torch.cat((pred_bbox1, pred_bbox2, pred_bbox3), dim=1)
        batch_detections, batch_annotations = get_detection_annotation(pred, targets, args.conf_thres, args.nms_thres, num_classes, args.img_size)
        val_detections += batch_detections
        val_annotations += batch_annotations
    
    with open(os.path.join(res_dir, f'val_detections.pkl'), 'wb') as f:
        pickle.dump(val_detections, f)
    with open(os.path.join(res_dir, f'val_annotations.pkl'), 'wb') as f:
        pickle.dump(val_annotations, f)
    # 以字典形式记录每一类的mAP值
    val_average_precisions = compute_mAP(val_detections, val_annotations, num_classes, args.iou_thres)

    # log everything and save model
    wandb.log({
        'mAP_train': train_average_precisions[0],
        'mAP_val': val_average_precisions[0],
        'loss': loss.item(),
        'loss_x_featmap1': np.mean(loss1_list['loss_x']),
        'loss_y_featmap1': np.mean(loss1_list['loss_y']),
        'loss_w_featmap1': np.mean(loss1_list['loss_w']),
        'loss_h_featmap1': np.mean(loss1_list['loss_h']),
        'loss_conf_featmap1': np.mean(loss1_list['loss_conf']),
        'loss_cls_featmap1': np.mean(loss1_list['loss_cls']),
        'loss_x_featmap2': np.mean(loss2_list['loss_x']),
        'loss_y_featmap2': np.mean(loss2_list['loss_y']),
        'loss_w_featmap2': np.mean(loss2_list['loss_w']),
        'loss_h_featmap2': np.mean(loss2_list['loss_h']),
        'loss_conf_featmap2': np.mean(loss2_list['loss_conf']),
        'loss_cls_featmap2': np.mean(loss2_list['loss_cls']),
        'loss_x_featmap3': np.mean(loss3_list['loss_x']),
        'loss_y_featmap3': np.mean(loss3_list['loss_y']),
        'loss_w_featmap3': np.mean(loss3_list['loss_w']),
        'loss_h_featmap3': np.mean(loss3_list['loss_h']),
        'loss_conf_featmap3': np.mean(loss3_list['loss_conf']),
        'loss_cls_featmap3': np.mean(loss3_list['loss_cls'])
    })
    if (epoch+1) % args.save_freq == 0:
        torch.save(model, os.path.join(model_dir, f'model-{epoch}.pth'))

model = torch.load(os.path.join(model_dir, f'model-{epoch}.pth'))
all_detections, all_annotations = [], []
img_path_res, detect_res, label_res = [], [], []
model.eval()
for img_path, imgs, targets in tqdm(val_dataloader):
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

with open(os.path.join(res_dir, f'all_detections_after.pkl'), 'wb') as f:
    pickle.dump(all_detections, f)
with open(os.path.join(res_dir, f'all_annotations_after.pkl'), 'wb') as f:
    pickle.dump(all_annotations, f)

# 以字典形式记录每一类的mAP值
average_precisions = compute_mAP(all_detections, all_annotations, num_classes, args.iou_thres)
print(f"mAP: {average_precisions[0]}")
