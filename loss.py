import torch
import torch.nn as nn
from utils import build_targets


class YOLOLayer(nn.Module):
    """
    这个类的作用有2个：
        1. 在测试和训练阶段，将网络输出的feature map转换成pred bbox、conf和类别
        2. 在训练阶段，另外一个作用是，根据feature map大小和anchor来build target，计算loss

    NOTE: 图片通道的顺序是CHW，输出的feature map通道的顺序是xywh、分数、类别概率
    """
    def __init__(self, anchors, num_classes, img_dim=(160, 320)):
        super().__init__()
        self.anchors = anchors # anchors = [(116,90),(156,198),(373,326)]
        self.num_classes = num_classes # 1
        self.image_dim = img_dim # (160, 320)
        self.ignore_thres = 0.5
        self.n_anchors = len(anchors) # 3
        self.bbox_attrs = 5 + num_classes

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, prediction, targets=None):
        # prediction: [1, 3, 13, 13, 85]
        # targets: [10, 5]
        n_batch, _, H, W, _ = prediction.size()
        stride = self.image_dim[0] / H # 416 / W = 416 / 13 = 32
        device = prediction.device

        x = torch.sigmoid(prediction[..., 0]) # center x: [1, 3, 13, 13]
        y = torch.sigmoid(prediction[..., 1]) # center y: [1, 3, 13, 13]
        w = prediction[..., 2] # width: [1, 3, 13, 13]
        h = prediction[..., 3] # height: [1, 3, 13, 13]
        pred_conf = torch.sigmoid(prediction[..., 4]) # [1, 3, 13, 13]
        pred_cls = torch.sigmoid(prediction[..., 5:]) # [1, 3, 13, 13, 80]

        # grid_x的shape为[1,1,nG,nG], 每一行的元素为:[0,1,2,3,...,nG-1]
        grid_x = torch.arange(W).repeat(H, 1).view([1, 1, H, W]).to(device)
        # grid_y的shape为[1,1,nG,nG], 每一列元素为: [0,1,2,3, ...,nG-1]
        grid_y = torch.arange(H).repeat(W, 1).t().view(1, 1, H, W).to(device)

        # scaled_anchors 是将原图上的 box 大小根据当前特征图谱的大小转换成相应的特征图谱上的 box
        # shape: [3, 2]
        scaled_anchors = torch.tensor([(a_h / stride, a_w / stride) for a_h, a_w in self.anchors]).to(device)

        # 分别获取其 w 和 h, 并将shape形状变为: [1,3,1,1]
        anchor_h = scaled_anchors[:, 0:1].view((1, self.n_anchors, 1, 1))
        anchor_w = scaled_anchors[:, 1:2].view((1, self.n_anchors, 1, 1))
        # shape: [1, 3, 13, 13, 4], 给 anchors 添加 offset 和 scale
        pred_boxes = torch.zeros(prediction[..., :4].shape).to(device)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        # 非训练阶段则直接返回预测结果, output的shape为: [n_batch, -1, 85]
        # [1, 3, 13, 13, 4]原本是3, 13, 13的维度，现在合并了
        output = torch.cat(
            (
                pred_boxes.view(n_batch, -1, 4) * stride, # 这里会对参数直接乘倍数，恢复正常
                pred_conf.view(n_batch, -1, 1),
                pred_cls.view(n_batch, -1, self.num_classes),
            ),
            -1,
        )
        if targets is None:
            return output
        else:  # 如果提供了 targets 标签, 则说明是处于训练阶段
            # 调用 utils.py 文件中的 build_targets 函数, 将真实的 box 数据转化成训练用的数据格式
            # mask: torch.Size([1, 3, 13, 13])
            # conf_mask: torch.Size([1, 3, 13, 13])
            # tx: torch.Size([1, 3, 13, 13])
            # ty: torch.Size([1, 3, 13, 13])
            # tw: torch.Size([1, 3, 13, 13])
            # th: torch.Size([1, 3, 13, 13])
            # tconf: torch.Size([1, 3, 13, 13])
            # tcls: torch.Size([1, 3, 13, 13, 80])
            mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=self.n_anchors,
                num_classes=self.num_classes,
                grid_size=(H, W),  # 这里说明，只有得到预测结果，才能build target
                ignore_thres=self.ignore_thres
            )

            # 处理 target Variables
            mask = mask.type(torch.bool)
            conf_mask = conf_mask.type(torch.bool)
            tx = tx.to(device)
            ty = ty.to(device)
            tw = tw.to(device)
            th = th.to(device)
            tconf = tconf.to(device)
            tcls = tcls.to(device)

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

            # 这里 conf_mask_true 指的是具有最佳匹配度的anchor box
            # conf_mask_false 指的是iou小于0.5的anchor box, 其余的anchor box都被忽略了
            loss_conf = self.bce_loss(
                pred_conf[conf_mask_false], tconf[conf_mask_false]
            ) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )

            # pred_cls[mask]的shape为: [7,80], torch.argmax(tcls[mask], 1)的shape为[7]
            # CrossEntropyLoss对象的输入为(x,class), 其中x为预测的每个类的概率, class为gt的类别下标
            loss_cls = (1 / n_batch) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))

            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item()
            ), output
