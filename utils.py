import torch 
import numpy as np
import math
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_single_detection_annotation(pred, targets, conf_thres=0.5, nms_thres=0.5, img_size=(160, 320)):
    """
    先用conf_thres和NMS过滤pred box，再把剩余的pred bbox和target bbox整理对齐，用于计算mAP

    Inputs:
        pred: shape=[B, N, 5], N是feature map所有的bbox
        targets: shape=[B, M, 5], M是这种图片上所有的GT bbox
    Returns:
        batch_detections: shape=[B, K, 5]
        batch_annotations: shape=[B, M, 4]
    """
    H, W = img_size
    outputs = nms_single_class(pred, conf_thres=conf_thres, nms_thres=nms_thres)

    batch_detections, batch_annotations = [], []
    for output, annotations in zip(outputs, targets):
        pred_boxes = np.array([])
        if output is not None:
            sort_i = np.argsort(output[:, 4])  # 按照置信度对bbox从小到大排序
            pred_boxes = output[sort_i]
        batch_detections.append(pred_boxes)

        annotation_boxes = np.array([])
        if any(annotations[:, -1] > 0):  # 但凡存在一个物体
            _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]
            # 将box的格式转换成x1,y1,x2,y2的形式, 同时将图片放缩至img_size大小
            annotation_boxes = np.empty_like(_annotation_boxes)
            annotation_boxes[:, 0] = (_annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2) * W
            annotation_boxes[:, 1] = (_annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2) * H
            annotation_boxes[:, 2] = (_annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2) * W
            annotation_boxes[:, 3] = (_annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2) * H
        batch_annotations.append(annotation_boxes)
    return batch_detections, batch_annotations


def get_detection_annotation(pred, targets, conf_thres=0.5, nms_thres=0.5, num_classes=1, img_size=(160, 320)):
    """
    输入pred bbox和GT bbox，它是一个batch的数据
    用conf_thres和NMS过滤pred box，得到按类别存放的pred bbox和GT bbox，用于计算mAP
    """
    H, W = img_size
    # 对计算结果执行 NMS 算法
    # outputs的shape为:[batch_size, m, 7]，每张图片保留m个pred bbox
    outputs = non_max_suppression(pred, num_classes, conf_thres=conf_thres, nms_thres=nms_thres)

    batch_detections, batch_annotations = [], []
    for output, annotations in zip(outputs, targets): # targets的shape为:[batch_size, n, 5]
        # 根据类别的数量创建占位空间, all_detections为一个列表, 列表中只有一个元素,
        # 该元素还是一个列表, 该列表中有80个np元素
        # 返回值中box的shape为: (x1, y1, x2, y2, object_conf, class_score, class_pred)
        one_detections = [np.array([]) for _ in range(num_classes)]
        if output is not None:
            # 获取预测结果的相应值
            pred_boxes = output[:, :5].cpu().numpy() # 坐标和包含物体的概率obj_conf
            scores = output[:, 4].cpu().numpy() # 置信度
            pred_labels = output[:, -1].cpu().numpy() # 类别编号
            # 按照置信度对预测的box进行排序，从小到大排序
            sort_i = np.argsort(scores)
            pred_labels = pred_labels[sort_i]
            pred_boxes = pred_boxes[sort_i]
            for label in range(num_classes):
                # all_detections是只有一个元素的列表, 因此这里用-1,
                # 获取所有预测类别为label的预测box, 可以将all_detections的shape看作为[1,1,80]
                one_detections[label] = pred_boxes[pred_labels == label]
                # 获取所有预测类别=label的box，这样就把1张图片中的pred bbox按照类别分组，每一类可以有多个bbox
        batch_detections.append(one_detections)

        # 上面把输出的pred bbox按照类别分组，每组按照conf排序；下面把label也按照类别分组；
        # 注意，怪不得要创建all_dections和all_annotations的[]，原来是为了对batch分别计算
        one_annotations = [np.array([]) for _ in range(num_classes)]
        if any(annotations[:, -1] > 0):  # 但凡存在一个物体
            annotation_labels = annotations[annotations[:, -1] > 0, 0].cpu().numpy() # 获取类别编号
            _annotation_boxes = annotations[annotations[:, -1] > 0, 1:].cpu().numpy() # 获取box坐标
            # 将box的格式转换成x1,y1,x2,y2的形式, 同时将图片放缩至opt.img_size大小
            # NOTE: label的保存格式是xywh
            annotation_boxes = np.empty_like(_annotation_boxes)
            annotation_boxes[:, 0] = (_annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2) * W
            annotation_boxes[:, 1] = (_annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2) * H
            annotation_boxes[:, 2] = (_annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2) * W
            annotation_boxes[:, 3] = (_annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2) * H
            # 因为原始的标签数据是以小数形式存储的, 所以可以直接利用乘法进行放缩
            for label in range(num_classes):
                one_annotations[label] = annotation_boxes[annotation_labels == label, :]
        batch_annotations.append(one_annotations)
    return batch_detections, batch_annotations


def compute_single_AP(all_detections, all_annotations, iou_thres=0.5):
    """
    Inputs:
        all_detections: shape=[N, K, 5]，每张图片有K个bbox，K可能=0
        all_annotations: shape=[N, M, 4]，每张图片有M个bbox，M可能=0
    Returns:
        average precision
    """
    true_positives = []
    scores = []
    num_annotations = 0
    # 遍历batch张图片的标注
    for i in range(len(all_annotations)):
        detections = all_detections[i]
        annotations = all_annotations[i]
        # 全部正例数量
        num_annotations += annotations.shape[0]
        detected_annotations = []
        # 遍历图片中的每个bbox
        for *bbox, score in detections:
            scores.append(score)

            if annotations.shape[0] == 0:
                true_positives.addpend(0) # 当前box并非真正例
                continue

            # 利用./utils/utils.py文件中的bbox_iou_numpy函数获取交并比矩阵(都是同类的box)
            overlaps = bbox_iou_numpy(np.array(bbox), annotations)
            assigned_annotation = np.argmax(overlaps) # 获取最大交并比的下标
            max_overlap = overlaps[assigned_annotation] # 获取最大交并比

            if max_overlap >= iou_thres and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                true_positives.append(0)

    # 如果没有物体出现在所有图片中, 在当前类的 AP 为 0
    if num_annotations == 0:
        AP = 0
    else:
        true_positives = np.array(true_positives) # 将列表转化成numpy数组
        false_positives = np.ones_like(true_positives) - true_positives

        #按照socre进行排序
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # 统计假正例和真正例
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # 计算召回率和准确率
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # 调用utils.py文件中的compute_ap函数计算average precision
        AP = compute_ap(recall, precision)
    return AP


def compute_mAP(all_detections, all_annotations, num_classes=1, iou_thres=0.5):
    """
    计算每一类别的AP指标
    """
    # 以字典形式记录每一类的mAP值
    average_precisions = {}
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0

        for i in range(len(all_annotations)):

            # 获取同类的预测结果和标签信息, i代表当前图片在batch中的位置
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []
            # 遍历该label的该batch下的每个bbox
            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.addpend(0) # 当前box并非真正例
                    continue

                # 利用./utils/utils.py文件中的bbox_iou_numpy函数获取交并比矩阵(都是同类的box)
                overlaps = bbox_iou_numpy(np.array(bbox), annotations)
                assigned_annotation = np.argmax(overlaps) # 获取最大交并比的下标
                max_overlap = overlaps[assigned_annotation] # 获取最大交并比

                if max_overlap >= iou_thres and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        # 如果当前类没有出现在该图片中, 在当前类的 AP 为 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives) # 将列表转化成numpy数组
        false_positives = np.ones_like(true_positives) - true_positives

        #按照socre进行排序
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # 统计假正例和真正例
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # 计算召回率和准确率
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # 调用utils.py文件中的compute_ap函数计算average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision
    return average_precisions


def nms_single_class(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    移除那些置信度低于conf_thres的boxes，在剩余的boxes上执行NMS算法。
    先选出具有最大score的box，删除与该box交并比大于阈值的box，接着继续选下一个最大socre的box, 重复上述操作，直至bbox为空。

    Inputs: 
        prediction: shape = (B, 2400, 5), 2400是feature map上anchor box的总数。
    Returns:
        output: shape = (B, N, 5)，N是每张图片剩余的bbox
    """
    # xywh->xyxy
    box_corner = np.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 先清除所有置信度小于conf_thres的box
        detections = image_pred[image_pred[:, 4] >= conf_thres]
        if not detections.shape[0]:
            continue

        # 按照每个box的置信度进行排序(第5维代表置信度 score)
        conf_sort_index = np.argsort(-detections[:, 4])
        detections = detections[conf_sort_index]
        max_detections = []
        while detections.shape[0]:
            # 将具有最大score的box添加到max_detections列表中,
            max_detections.append(detections[0])
            # 当只剩下一个box时, 当前类的nms过程终止
            if len(detections) == 1:
                break
            # 获取当前最大socre的box与其余同类box的iou, 调用了本文件的bbox_iou()函数
            ious = bbox_iou_numpy(max_detections[-1], detections[1:])
            # 移除那些交并比大于阈值的box(也即只保留交并比小于阈值的box)
            detections = detections[1:][ious < nms_thres]
        # 将执行nms后的剩余的同类box连接起来, 最终shape为[m, 5], m为nms后同类box的数量
        max_detections = np.stack(max_detections)
        # 将计算结果添加到output返回值当中, output是一个列表, 列表中的每个元素代表这一张图片的nms后的box
        output[image_i] = max_detections 
    return output


# nms: 对于每一类(不同类之间的box不执行nms), 先选出具有最大score的box, 删除与该box交并比较大的同类box,
# 接着继续选下一个最大socre的box, 直至同类box为空, 然后对下一类执行nms
# 注意yolo与faster rcnn在执行nms算法时的不同, 前者是在多类上执行的, 后者是在两类上执行的
def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # prediction的shape为: [1,10647,85], 其中, 1为batch_size, 10647是尺寸为416的图片的anchor box的总数
    # 移除那些置信度低于conf_thres的boxes, 同时在剩余的boxes上执行NMS算法
    # 返回值中box的shape为: (x1, y1, x2, y2, object_conf, class_score, class_pred)

    # xywh->xyxy
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    # len(prediction)为Batch_size, 这里申请了占位空间, 大小为batch_size
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 先清除所有置信度小于conf_thres的box, conf_mask的shape为:[n], n为置信度大于阈值的box数量
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze() # 这里的squeeze()可加可不加
        image_pred = image_pred[conf_mask] # image_pred的shape为[n, 85]

        if not image_pred.size(0):
            continue # 如果所有的box的置信度都小于阈值, 那么就跳过当前的图片, 对下一张进行操作

        # 获取每个box的类别的预测结果和编号(0~79), 使用了keepdim, 否则shape维数会减一(dim指定的维度会消失)
        # class_conf的shape为[n, 1], 代表n个box的score
        # class_pred的shape为[n, 1], 代表n个box的类别编号
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)

        # 对以上结果进行汇总, shape为[n,7]: (x1,y1,x2,y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # 获取当前image中出现过的类别号, 然后分别对每一类执行NMS算法
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        # 分别对每一类执行NMS算法, 注意这点与faster rcnn不同, 后者只对两类执行nms算法, 也就是是否出现物体
        # faster rcnn的nms算法会有一个问题, 那就是当两个不同物体重复度较高时, fasterrcnn会忽略置信度较低的一个
        for c in unique_labels:
            # 获取指定类别的所有box
            detections_class = detections[detections[:, -1] == c] # detections的最后一维指示类别编号

            # 按照每个box的置信度进行排序(第5维代表置信度 score)
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]

            # 执行NMS算法, 核心思想是先将具有最大socre的box放置在max_detections列表当中,
            # 然后令该box与剩余的所有同类box计算交并比, 接着删除那些具有较大交并比的box(大于阈值)
            # 重复对detections_class执行上面两步操作, 知道detections_class中只剩下一个box为止
            max_detections = []
            while detections_class.size(0):
                # 将具有最大score的box添加到max_detections列表中,
                # 注意要将box的shape扩展成:[1,7], 方便后续max的连接(cat)
                max_detections.append(detections_class[0].unsqueeze(0))

                # 当只剩下一个box时, 当前类的nms过程终止
                if len(detections_class) == 1:
                    break

                # 获取当前最大socre的box与其余同类box的iou, 调用了本文件的bbox_iou()函数
                ious = bbox_iou(max_detections[-1], detections_class[1:])

                # 移除那些交并比大于阈值的box(也即只保留交并比小于阈值的box)
                detections_class = detections_class[1:][ious < nms_thres]

            # 将执行nms后的剩余的同类box连接起来, 最终shape为[m, 7], m为nms后同类box的数量
            max_detections = torch.cat(max_detections).data

            # 将计算结果添加到output返回值当中, output是一个列表, 列表中的每个元素代表这一张图片的nms后的box
            # 注意, 此时同一张图片的不同类的box也会连接到一起, box的最后一维会存储类别编号(4+1+1+1).
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat(
                    (output[image_i], max_detections)
                )
            )

    return output


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    # 返回 box1 和 box2 的 iou, box1 和 box2 的 shape 要么相同, 要么其中一个为[1,4]
    if not x1y1x2y2:
        # 获取 box1 和 box2 的左上角和右下角坐标
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # 获取 box1 和 box2 的左上角和右下角坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 获取相交矩形的左上角和右下角坐标
    # 注意, torch.max 函数要求输入的两个参数要么 shape 相同, 此时在相同位置上进行比较并取最大值
    # 要么其中一个 shape 的第一维为 1, 此时会自动将该为元素与另一个 box 的所有元素做比较, 这里使用的就是该用法.
    # 具体来说, b1_x1 为 [1, 1], b2_x1 为 [3, 1], 此时会有 b1_x1 中的一条数据分别与 b2_x1 中的三条数据做比较并取最大值
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # 计算相交矩形的面积
    inter_width = torch.clamp(inter_rect_x2-inter_rect_x1, min=0)
    inter_height = torch.clamp(inter_rect_y2-inter_rect_y1, min=0)
    inter_area = inter_width * inter_height
    # 分别求 box1 矩形和 box2 矩形的面积.
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # 计算 iou 并将其返回
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def bbox_iou_numpy(rect1, rectangles, x1y1x2y2=True):
    # 返回 box1 和 box2 的 iou, box1 和 box2 的 shape 要么相同, 要么为(4,) 和 (B,4)
    if not x1y1x2y2:
        # 获取 box1 和 box2 的左上角和右下角坐标
        rect1 = np.concatenate([rect1[:2]-rect1[2:]/2, rect1[:2]+rect1[2:]/2])
        rectangles = np.concatenate(
            [rectangles[:, :2] - rectangles[:, 2:] / 2,
             rectangles[:, :2] + rectangles[:, 2:] / 2], axis=-1
        )

    # 计算交集区域的左上角坐标
    x_intersection = np.maximum(rect1[0], rectangles[:, 0])
    y_intersection = np.maximum(rect1[1], rectangles[:, 1])
    
    # 计算交集区域的右下角坐标
    x_intersection_end = np.minimum(rect1[2], rectangles[:, 2])
    y_intersection_end = np.minimum(rect1[3], rectangles[:, 3])
    
    # 计算交集区域的宽度和高度（可能为负数，表示没有重叠）
    intersection_width = np.maximum(0, x_intersection_end - x_intersection)
    intersection_height = np.maximum(0, y_intersection_end - y_intersection)
    
    # 计算交集区域的面积
    intersection_area = intersection_width * intersection_height
    
    # 计算矩形1的面积
    area_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    
    # 计算其他矩形的面积
    area_rectangles = (rectangles[:, 2] - rectangles[:, 0]) * (rectangles[:, 3] - rectangles[:, 1])
    
    # 计算并集区域的面积
    union_area = area_rect1 + area_rectangles - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area
    
    return iou


def build_targets(target, anchors, num_anchors, grid_size, ignore_thres):
    # 参数:
    # target: [1, 50, 5]
    # anchors: [3, 2]
    # num_anchors: 3
    # num_classes: 80
    # grid_size: 13(特征图谱的尺寸)
    # conf_thres: 0.5
    # img_dim: 图片尺寸
    nB = target.size(0) # batch_size
    nA = num_anchors # 3
    H, W = grid_size # 特征图谱的尺寸(eg: 13)
    mask = torch.zeros(nB, nA, H, W) # eg: [1, 3, 13, 13], 代表每个特征图谱上的 anchors 下标(每个 location 都有 3 个 anchors)
    conf_mask = torch.ones(nB, nA, H, W) # eg: [1, 3, 13, 13] 代表每个 anchor 的置信度.
    tx = torch.zeros(nB, nA, H, W) # 申请占位空间, 存放每个 anchor 的中心坐标
    ty = torch.zeros(nB, nA, H, W) # 申请占位空间, 存放每个 anchor 的中心坐标
    tw = torch.zeros(nB, nA, H, W) # 申请占位空间, 存放每个 anchor 的宽
    th = torch.zeros(nB, nA, H, W) # 申请占位空间, 存放每个 anchor 的高
    tconf = torch.zeros(nB, nA, H, W) # 占位空间, 存放置信度, eg: [1, 3, 13, 13]

    # 对每张图片
    for b in range(nB):
        # 对每个GT bbox
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0: # b指定的batch中的某图片, t指定了图片中的某 box(按顺序)
                continue # 如果 box 的5个值(从标签到坐标)都为0, 那么就跳过当前的 box

            # Convert to position relative to box
            # 由于我们在存储box的坐标时, 就是按照其相对于图片的宽和高的比例存储的
            # 因此, 当想要获取特征图谱上的对应 box 的坐标时, 直接令其与特征图谱的尺寸相乘即可.
            gx = target[b, t, 1] * W
            gy = target[b, t, 2] * H
            gw = target[b, t, 3] * W
            gh = target[b, t, 4] * H

            # Get grid box indices
            # 获取在特征图谱上的整数坐标
            gi = int(gx)
            gj = int(gy)

            # Get shape of gt box, 根据 box 的大小获取 shape: [1,4]
            gt_box = torch.tensor(np.array([0, 0, gw, gh])).unsqueeze(0)

            # Get shape of anchor box
            # 相似的方法得到anchor的shape: [3, 4] , 3 代表3个anchor
            anchor_shapes = torch.tensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))

            # 调用本文件的 bbox_iou 函数计算gt_box和anchors之间的交并比
            # 注意这里仅仅计算的是 shape 的交并比, 此处没有考虑位置关系.
            # gt_box 为 [1,4], anchors 为 [3, 4],
            # 最终返回的值为[3], 代表了 gt_box 与每个 anchor 的交并比大小
            # NOTE: 这里是xywh的格式
            anch_ious = bbox_iou(gt_box, anchor_shapes, x1y1x2y2=False)

            # 将交并比大于阈值的部分设置conf_mask的对应位为0(ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0

            # 找到匹配度最高的 anchor box, 返回下标: 0,1,2 中的一个
            best_n = np.argmax(anch_ious)

            # 设置 mask 和 conf_mask
            mask[b, best_n, gj, gi] = 1
            # 注意, 刚刚将所有大于阈值的 conf_mask对应为都设置为了0,
            # 然后这里将具有最大交并比的anchor设置为1, 如此确保一个真实框只对应一个 anchor.
            # 由于 conf_mask 的默认值为1, 因此, 剩余的box可看做是负样本
            conf_mask[b, best_n, gj, gi] = 1

            # 设置中心坐标, 该坐标是相对于 cell的左上角而言的, 所以是一个小于1的数
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # 设置宽和高, 注意, 这里会转化成训练时使用的宽高值
            # NOTE: 这里anchor是H, W的顺序，但网络输出按照W, H的顺序
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][1] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][0] + 1e-16)

            # 将置信度设置为 1
            tconf[b, best_n, gj, gi] = 1

    # 将所有需要的信息都返回, 从这里可以看出, 每一个 YOLO 层都会执行一次预测.
    return mask, conf_mask, tx, ty, tw, th, tconf
