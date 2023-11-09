import os
import argparse
import torch
import time
from tqdm import tqdm
import wandb
import numpy as np

from model import MODEL_REGISTRY
from dataset import ListDataset
from loss import YOLOLayer
from utils import set_seed, get_single_cls_detection_annotation, compute_single_cls_ap

# TODO: 两种模型数据集不统一
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='YOLOv3')
    parser.add_argument("--model", type=str, default='Darknet53')
    parser.add_argument("--device", type=int, default=0, help="cpu if <0, or gpu id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--data_dir", type=str, default="/home/agent/Code/ackermann_car_nav/data/rss_mask", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="path to results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--init_filter", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--conf_thres", type=float, default=0.5, help="objectiveness confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, nargs='+', default=[160, 320])
    parser.add_argument("--save_freq", type=int, default=20)
    args = parser.parse_args()
    return args


def train(args):
    model = MODEL_REGISTRY[args.model](args.init_filter).to(args.device)

    train_dataloader = torch.utils.data.DataLoader(
        ListDataset(args.data_dir, mode='train'), batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        ListDataset(args.data_dir, mode='val'), batch_size=args.batch_size
    )
    anchors = torch.tensor([[10, 13], [16, 30], [33, 23]])

    YOLOLoss = YOLOLayer(anchors, img_dim=args.img_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.epochs)):
        train_loss_list, train_mse_loss_list, train_conf_loss_list, train_ciou_loss_list = [], [], [], []
        train_annotations, train_detections = [], []
        model.train()
        for _, imgs, masks, targets in train_dataloader:
            imgs = imgs.to(args.device)
            masks = masks.to(args.device)
            targets = targets.to(args.device)
            # y = model(imgs, masks)
            y = model(imgs)
            loss_dict, pred_bbox = YOLOLoss(y, targets)
            train_loss = loss_dict['loss']
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss_list.append(train_loss.item())
            train_mse_loss_list.append(loss_dict['loss_mse'])
            train_conf_loss_list.append(loss_dict['loss_conf'])
            train_ciou_loss_list.append(loss_dict['loss_ciou'])

            batch_detections, batch_annotations = get_single_cls_detection_annotation(
                pred_bbox.data.cpu().numpy(), targets.data.cpu().numpy(), 
                args.conf_thres, args.nms_thres, args.img_size
            )
            train_detections += batch_detections
            train_annotations += batch_annotations
        train_AP50 = compute_single_cls_ap(train_detections, train_annotations)

        val_loss_list, val_mse_loss_list, val_conf_loss_list, val_ciou_loss_list = [], [], [], []
        val_annotations, val_detections = [], []
        model.eval()
        for _, imgs, masks, targets in val_dataloader:
            imgs = imgs.to(args.device)
            masks = masks.to(args.device)
            targets = targets.to(args.device)
            with torch.no_grad():
                # y = model(imgs, masks)
                y = model(imgs)
                loss_dict, pred_bbox = YOLOLoss(y, targets)
            val_loss = loss_dict['loss']
            val_loss_list.append(val_loss.item())
            val_mse_loss_list.append(loss_dict['loss_mse'])
            val_conf_loss_list.append(loss_dict['loss_conf'])
            val_ciou_loss_list.append(loss_dict['loss_ciou'])

            batch_detections, batch_annotations = get_single_cls_detection_annotation(
                pred_bbox.data.cpu().numpy(), targets.data.cpu().numpy(), 
                args.conf_thres, args.nms_thres, args.img_size
            )
            val_detections += batch_detections
            val_annotations += batch_annotations
        val_AP50 = compute_single_cls_ap(val_detections, val_annotations)

        wandb.log({
            'mAP_train': train_AP50,
            'mAP_val': val_AP50,
            'train_loss': np.mean(train_loss_list),
            'train_mse_loss': np.mean(train_mse_loss_list),
            'train_conf_loss': np.mean(train_conf_loss_list),
            'train_ciou_loss': np.mean(train_ciou_loss_list),
            'val_loss': np.mean(val_loss_list),
            'val_mse_loss': np.mean(val_mse_loss_list),
            'val_conf_loss': np.mean(val_conf_loss_list),
            'val_ciou_loss': np.mean(val_ciou_loss_list),
            'lr': optimizer.param_groups[0]['lr']
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
