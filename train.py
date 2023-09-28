import os
import argparse
import torch
import time
from torchsummary import summary
from tqdm import tqdm
import wandb
import numpy as np

from model import Darknet
from dataset import ListDataset
from loss import YOLOLayer
from utils import set_seed, get_detection_annotation, compute_mAP


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='YOLOv3')
    parser.add_argument("--device", type=int, default=0, help="cpu if <0, or gpu id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--data_dir", type=str, default="/home/agent/Code/datasets/data_20230626_parallel", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="path to results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="objectiveness confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, nargs='+', default=[160, 320])
    parser.add_argument("--save_freq", type=int, default=20)
    args = parser.parse_args()
    return args


def train(args):
    num_classes = 1
    init_filter = 8
    model = Darknet(num_classes, init_filter).to(args.device)
    summary(model, (3, 160, 320))

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
    Loss1 = YOLOLayer(anchors[2], num_classes, img_dim=args.img_size)
    Loss2 = YOLOLayer(anchors[1], num_classes, img_dim=args.img_size)
    Loss3 = YOLOLayer(anchors[0], num_classes, img_dim=args.img_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
        for _, imgs, targets in train_dataloader:
            # imgs: [B, 3, 416, 416]
            # targets: [B, 50, 5]
            optimizer.zero_grad()
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)
            y1, y2, y3 = model(imgs)
            loss1_dict, pred_bbox1 = Loss1(y1, targets)
            loss2_dict, pred_bbox2 = Loss2(y2, targets)
            loss3_dict, pred_bbox3 = Loss3(y3, targets)
            loss = loss1_dict[0] + loss2_dict[0] + loss3_dict[0]
            loss.backward()
            optimizer.step()

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
        train_average_precisions = compute_mAP(train_detections, train_annotations, num_classes, args.iou_thres)

        model.eval()
        val_annotations, val_detections = [], []
        for _, imgs, targets in val_dataloader:
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)
            with torch.no_grad():
                y1, y2, y3 = model(imgs)
                pred_bbox1 = Loss1(y1)
                pred_bbox2 = Loss2(y2)
                pred_bbox3 = Loss3(y3)
                pred = torch.cat((pred_bbox1, pred_bbox2, pred_bbox3), dim=1)
            batch_detections, batch_annotations = get_detection_annotation(pred, targets, args.conf_thres, args.nms_thres, num_classes, args.img_size)
            val_detections += batch_detections
            val_annotations += batch_annotations
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
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'model-{epoch}.pth'))


if __name__ == '__main__':
    args = get_args()

    set_seed(args.seed)
    args.device = f'cuda:{args.device}' if args.device >=0 and torch.cuda.is_available() else 'cpu'
    logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = os.path.join(args.output_dir, logid)
    args.model_dir = os.path.join(log_dir, 'model')
    args.res_dir = os.path.join(log_dir, 'result')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.res_dir, exist_ok=True)
    wandb.init(
        project=f"{args.project_name}",
        name=f"{logid}",
        config=vars(args)
    )

    train(args)
