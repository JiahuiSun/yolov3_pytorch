import torch
import torch.nn as nn
import math

from utils import build_targets


def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:

    举例时假设pred_xywh和label_xywh的shape都是(1, 4)
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = torch.cat((boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5), dim=-1)
    boxes2_x0y0x1y1 = torch.cat((boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5), dim=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = torch.cat((torch.min(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                             torch.max(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])), dim=-1)
    boxes2_x0y0x1y1 = torch.cat((torch.min(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                             torch.max(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])), dim=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = torch.max(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = torch.min(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = right_down - left_up
    inter_section = torch.where(inter_section < 0.0, inter_section*0, inter_section)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = torch.min(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = torch.max(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = torch.pow(enclose_wh[..., 0], 2) + torch.pow(enclose_wh[..., 1], 2)

    # 两矩形中心点距离的平方
    p2 = torch.pow(boxes1[..., 0] - boxes2[..., 0], 2) + torch.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # 增加av。加上除0保护防止nan。
    atan1 = torch.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-9))
    atan2 = torch.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-9))
    v = 4.0 * torch.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou


def CIOULoss(pred_bboxes, txywh, img_size=(160, 320)):
    ciou = bbox_ciou(pred_bboxes, txywh)  # N
    area = img_size[0] * img_size[1]

    # 每个预测框xxxiou_loss的权重 = 2 - (ground truth的面积/图片面积)
    bbox_loss_scale = 2.0 - txywh[:, 2] * txywh[:, 3] / area
    ciou_loss = bbox_loss_scale * (1 - ciou)
    return ciou_loss.mean()


class YOLOLayer(nn.Module):
    """
    这个类的作用有2个：
        1. 在测试和训练阶段，将网络输出的feature map转换成pred bbox、conf和类别
        2. 在训练阶段，另外一个作用是，根据feature map大小和anchor来build target，计算loss

    NOTE: 图片通道的顺序是CHW，anchor的顺序是HW，输出的feature map通道的顺序是xywh、分数、类别概率
    """
    def __init__(self, anchors, img_dim=(160, 320)):
        super().__init__()
        self.anchors = anchors # anchors = [(116,90),(156,198),(373,326)]
        self.img_dim = img_dim # (160, 320)

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.ciou_loss = CIOULoss

    def forward(self, prediction, targets=None):
        # prediction: [1, 3, 13, 13, 85]
        # targets: [10, 5]
        n_batch, n_anchor, H, W, n_dim = prediction.size()
        stride = self.img_dim[0] / H # 416 / W = 416 / 13 = 32
        device = prediction.device

        x = torch.sigmoid(prediction[..., 0]) # center x: [1, 3, 13, 13]
        y = torch.sigmoid(prediction[..., 1]) # center y: [1, 3, 13, 13]
        w = prediction[..., 2] # width: [1, 3, 13, 13]
        h = prediction[..., 3] # height: [1, 3, 13, 13]
        pred_conf = torch.sigmoid(prediction[..., 4]) # [1, 3, 13, 13]

        # grid_x的shape为[1,1,nG,nG], 每一行的元素为:[0,1,2,3,...,nG-1]
        grid_x = torch.arange(W).repeat(H, 1).view([1, 1, H, W]).to(device)
        # grid_y的shape为[1,1,nG,nG], 每一列元素为: [0,1,2,3, ...,nG-1]
        grid_y = torch.arange(H).repeat(W, 1).t().view(1, 1, H, W).to(device)

        # scaled_anchors 是将原图上的 box 大小根据当前特征图谱的大小转换成相应的特征图谱上的 box, shape: [3, 2]
        scaled_anchors = torch.tensor([(a_h / stride, a_w / stride) for a_h, a_w in self.anchors]).to(device)

        # 分别获取其 w 和 h, 并将shape形状变为: [1,3,1,1]
        anchor_h = scaled_anchors[:, 0:1].view((1, self.anchors.size(0), 1, 1))
        anchor_w = scaled_anchors[:, 1:2].view((1, self.anchors.size(0), 1, 1))
        # shape: [1, 3, 13, 13, 4], 给 anchors 添加 offset 和 scale
        pred_bboxes = torch.stack(
            [x+grid_x, y+grid_y, torch.exp(w)*anchor_w, torch.exp(h)*anchor_h], dim=-1
        ) * stride  # 这里对参数直接乘倍数，恢复正常

        # 非训练阶段则直接返回预测结果, output的shape为: [n_batch, -1, 85]
        output = torch.cat(
            [pred_bboxes.view(n_batch, -1, 4), pred_conf.view(n_batch, -1, 1)], dim=-1
        )
        if targets is None:
            return output
        # 如果提供了 targets 标签, 则说明是处于训练阶段
        # 调用 utils.py 文件中的 build_targets 函数, 将真实的 box 数据转化成训练用的数据格式
        # mask: torch.Size([1, 3, 13, 13])
        # conf_mask: torch.Size([1, 3, 13, 13])
        # tx: torch.Size([1, 3, 13, 13])
        # ty: torch.Size([1, 3, 13, 13])
        # tw: torch.Size([1, 3, 13, 13])
        # th: torch.Size([1, 3, 13, 13])
        # tconf: torch.Size([1, 3, 13, 13])
        # txywh: (B, 3, 13, 13, 4)
        mask, conf_mask, tx, ty, tw, th, tconf, txywh = build_targets(
            target=targets.data.cpu(),
            anchors=scaled_anchors.data.cpu(),
            grid_size=(H, W),  # 这里说明，只有得到feature map，才能build target
            img_size=self.img_dim
        )

        mask = mask.type(torch.bool)
        conf_mask = conf_mask.type(torch.bool)
        tx = tx.to(device)
        ty = ty.to(device)
        tw = tw.to(device)
        th = th.to(device)
        tconf = tconf.to(device)
        txywh = txywh.to(device)

        # 获取表明gt和非gt的conf_mask
        # 这里 conf_mask_true 指的是具有最佳匹配度的anchor box
        # conf_mask_false 指的是iou小于0.5的anchor box, 其余的anchor box都被忽略了
        conf_mask_true = mask # mask 只有best_n对应位为1, 其余都为0
        conf_mask_false = conf_mask^mask # conf_mask中iou大于ignore_thres的为0, 其余为1, best_n也为1

        # 忽略 non-existing objects, 计算相应的loss
        loss_x = self.mse_loss(x[mask], tx[mask])
        loss_y = self.mse_loss(y[mask], ty[mask])
        loss_w = self.mse_loss(w[mask], tw[mask])
        loss_h = self.mse_loss(h[mask], th[mask])
        loss_xywh = loss_x + loss_y + loss_w + loss_h

        # ciou loss
        loss_ciou = self.ciou_loss(pred_bboxes[mask], txywh[mask], self.img_dim)

        # 这里 conf_mask_true 指的是具有最佳匹配度的anchor box
        # conf_mask_false 指的是iou小于0.5的anchor box, 其余的anchor box都被忽略了
        loss_conf = self.bce_loss(
            pred_conf[conf_mask_false], tconf[conf_mask_false]
        ) + self.bce_loss(
            pred_conf[conf_mask_true], tconf[conf_mask_true]
        )

        loss = loss_ciou + loss_conf

        return (
            loss,
            loss_xywh.item(),
            loss_conf.item(),
            loss_ciou.item()
        ), output
