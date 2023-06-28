import numpy as np
import torch

def cal_iou(pre_mask,gt_mask):
    intersection = torch.sum(pre_mask * gt_mask)
    union = torch.sum(pre_mask) + torch.sum(gt_mask) - intersection
    
    if union.item() == 0:
        iou = torch.tensor(0, device='cuda')
        dice = torch.tensor(0, device='cuda')
    else:
        iou = intersection / union
        dice = (2.0 * intersection) / (torch.sum(pre_mask) + torch.sum(gt_mask))
    return iou , dice