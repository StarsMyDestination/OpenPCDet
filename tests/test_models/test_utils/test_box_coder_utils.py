import numpy as np
import torch
from scipy.stats import special_ortho_group
from pcdet.utils.box_coder_utils import PointResidualCoderAzimuthInvariant


def test_pointResidualCoderAzimuthInvariant():
    points = np.random.random((100, 3))
    R = special_ortho_group.rvs(3, size=100, random_state=1024)
    theta = (np.random.random((100, 1)) - 0.5) * 2 * np.pi
    rotation_matrix = np.hstack([R.reshape(-1, 9), theta])

    gt_boxes = np.random.random((100, 7))
    gt_classes = np.random.randint(0, 2, (100, 1))
    gt_boxes = np.hstack([gt_boxes, gt_classes])
    mean_size = [[3.9, 1.6, 1.56],
                 [0.8, 0.6, 1.73],
                 [1.76, 0.6, 1.73]]

    points = torch.from_numpy(points)
    rotation_matrix = torch.from_numpy(rotation_matrix)
    gt_boxes = torch.from_numpy(gt_boxes)

    # ==== no mean_size test ==========#
    box_coder = PointResidualCoderAzimuthInvariant(use_mean_size=False)

    box_encoding = box_coder.encode_torch(gt_boxes, points, rotation_matrix)

    box_decoded = box_coder.decode_torch(box_encoding, points, rotation_matrix)

    delta = (box_decoded - gt_boxes).sum()
    print('gt_boxes: {}'.format(gt_boxes))
    print('box_decoded: {}'.format(box_decoded))
    print(delta)
    assert delta < 1e-5

    # ==== use mean_size test ==========#
    box_coder = PointResidualCoderAzimuthInvariant(use_mean_size=True, mean_size=mean_size, cuda=False)

    box_encoding = box_coder.encode_torch(gt_boxes, points, rotation_matrix, gt_classes=gt_boxes[:, -1].long())

    box_decoded = box_coder.decode_torch(box_encoding, points, rotation_matrix, pred_classes=gt_boxes[:, -1].long())

    delta = (box_decoded - gt_boxes).sum()
    print('gt_boxes: {}'.format(gt_boxes))
    print('box_decoded: {}'.format(box_decoded))
    print(delta)
    assert delta < 1e-5


if __name__ == '__main__':
    test_pointResidualCoderAzimuthInvariant()
