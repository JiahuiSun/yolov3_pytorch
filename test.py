import numpy as np
import argparse
from tqdm import tqdm
import time
import os
import torch
import cv2

from model import Darknet
from loss import YOLOLayer
from utils import set_seed, compute_single_AP, get_single_detection_annotation, nms_single_class
from dataset import ListDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="cpu if <0, or gpu id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="output/20230929_112026/model/model-99.pth", help="path to model")
    parser.add_argument("--data_dir", type=str, default="/home/agent/Code/datasets/data_20230626_parallel", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="path to results")
    parser.add_argument("--init_filter", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="objectiveness confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, nargs='+', default=[160, 320])
    args = parser.parse_args()
    return args


def test(args):
    model = torch.load(args.model_path, map_location=args.device)

    dataloader = torch.utils.data.DataLoader(
        ListDataset(args.data_dir, mode='val'), batch_size=args.batch_size
    )
    anchors = torch.tensor([[10, 13], [16, 30], [33, 23]])

    YOLOLoss = YOLOLayer(anchors, img_dim=args.img_size)

    val_loss_list = []
    all_detections, all_annotations = [], []
    img_path_res, detect_res = [], []
    model.eval()
    for img_path, imgs, targets in tqdm(dataloader):
        imgs = imgs.to(args.device)
        targets = targets.to(args.device)
        with torch.no_grad():
            y = model(imgs)
            loss_dict, pred_bbox = YOLOLoss(y, targets)
        val_loss_list.append(loss_dict[0].item())

        # 为了画图
        img_path_res += img_path
        detections = nms_single_class(pred_bbox.clone().cpu().numpy(), args.conf_thres, args.nms_thres)
        detect_res += detections

        # 为了计算mAP
        batch_detections, batch_annotations = get_single_detection_annotation(pred_bbox.cpu().numpy(), targets.cpu().numpy(), args.conf_thres, args.nms_thres, args.img_size)
        all_detections += batch_detections
        all_annotations += batch_annotations
    average_precisions = compute_single_AP(all_detections, all_annotations, args.iou_thres)
    print(f"mAP: {average_precisions:.3f}, loss: {np.mean(val_loss_list):.3f}")

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
            for x1, y1, x2, y2, conf in detections:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color_pred, box_thick)
        
        out_path = os.path.join(args.res_dir, img_path.split('/')[-1])
        cv2.imwrite(out_path, img)


if __name__ == '__main__':
    args = get_args()

    set_seed(args.seed)
    args.device = f'cuda:{args.device}' if args.device >=0 and torch.cuda.is_available() else 'cpu'
    logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = os.path.join(args.output_dir, logid)
    args.res_dir = os.path.join(log_dir, 'result')
    os.makedirs(args.res_dir, exist_ok=True)

    test(args)
