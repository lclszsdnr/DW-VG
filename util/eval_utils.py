import torch
import numpy as np

from util.box_ops import bbox_iou
from util.box_ops import box_cxcywh_to_xyxy as  xywh2xyxy


def trans_vg_eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    return iou, accu

def trans_vg_eval_test(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)

    return accu_num



def get_pos_weight(tensor):
    total_example = torch.numel(tensor)
    num_positive_examples = torch.count_nonzero(tensor)
    pos_weight = (total_example - num_positive_examples) / (num_positive_examples+0.1)

    return pos_weight
