import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

from . import rangeview_cuda


class Point2RangeFunction(Function):
    @staticmethod
    def forward(ctx, points, pxpy, rv_hw):
        """
        Args:
            ctx:
            points: (N1+N2+..., C) points feature
            pxpy: (N, 3) batch_idx + points normed coords in range image, [-1, 1)
            rv_hw: tuple, range image size (H, W)
        Returns:
            range image: (B, C, H, W)
        """
        H, W = rv_hw
        N, C = points.shape
        B = pxpy[:, 0].max().int().item() + 1
        rv_image = points.new_zeros((B, C, H, W))

        coords = pxpy.new_zeros((N, 2))
        coords[:, 0] = ((pxpy[:, 1] + 1) / 2. * W).int()
        coords[:, 1] = ((pxpy[:, 2] + 1) / 2. * H).int()

        coords = torch.cat([pxpy[:, 0:1], coords], dim=-1).int()  # Nx3

        counts = pxpy.new_zeros((B, H, W), dtype=torch.int32)
        rangeview_cuda.map_count(coords, counts)

        rangeview_cuda.point2range(points, coords, counts, rv_image)

        ctx.for_backward = (counts, coords, N, C)
        return rv_image

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param grad_out: (B, C, H, W)
        :return:
            grad_in: (N, C)
        """
        counts, coords, N, C = ctx.for_backward

        grad_in = grad_out.new_zeros((N, C))
        rangeview_cuda.point2range_backward(grad_out.contiguous(), coords, counts, grad_in)

        return grad_in, None, None


def point_to_range(points, pxpy, rv_hw):
    """
    Args:
        ctx:
        points: (N1+N2+..., C) points feature
        pxpy: (N, 3) batch_idx + points normed coords in range image, [-1, 1)
        rv_hw: tuple, range image size (h, w)
    Returns:
        range image: (B, C, H, W)
    """
    rv_images = Point2RangeFunction.apply(points, pxpy, rv_hw)
    return rv_images


def range_to_point(rv_images, pxpy, use_xyz=False, xyz=None, mode='bilinear'):
    '''
    Args:
        rv_images: (B, C, H, W)
        pxpy: Nx3, 3=(bs_idx, px, py)
        xyz: Nx3
    Returns:
        points feature: (N, C)
    '''

    bs = rv_images.shape[0]
    feature_list = []
    for bs_idx in range(bs):
        mask = (pxpy[:, 0] == bs_idx)
        pxpy_single = pxpy[mask][:, 1:3]  # Nx2
        pxpy_single = pxpy_single.unsqueeze(0).unsqueeze(0)  # 1x1xNx2
        sampled_features = F.grid_sample(rv_images[bs_idx:(bs_idx + 1)], pxpy_single, mode=mode)  # 1xCx1xN
        sampled_features = sampled_features.squeeze(0).squeeze(1).transpose(1, 0).contiguous()  # NxC
        if use_xyz:
            assert xyz is not None
            sampled_features = torch.cat([xyz[mask], sampled_features], dim=-1)
        feature_list.append(sampled_features)

    return torch.cat(feature_list, dim=0)  # (N1 + N2 + N3 + ..., C)


if __name__ == '__main__':
    pass
