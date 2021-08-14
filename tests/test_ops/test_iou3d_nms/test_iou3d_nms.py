import numpy as np
import torch
from pcdet.ops.iou3d_nms import iou3d_nms_utils

if __name__ == '__main__':
    boxes_a = torch.tensor([
        [0, 0, 0, 4, 2, 1.5, 0],
        [10, 10, 1, 4, 2, 1.5, 0],
    ]).cuda()

    boxes_b = torch.tensor([
        [0, 0, 0, 4, 2, 1.5, 0],
        [12, 10, 1, 4, 2, 1.5, 1.5],
    ]).cuda()

    ious = iou3d_nms_utils.pair_wise_boxes_iou3d_gpu(boxes_a, boxes_b)
    print(ious.shape)
    print(ious)
