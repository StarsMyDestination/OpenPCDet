import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from pcdet.ops.rangeview import range_to_point, point_to_range

if __name__ == '__main__':
    N = 4
    points = torch.linspace(1, N, N).reshape(-1, 1).float()  # [1,2,...,N]
    pxpy = (torch.rand((N, 2)) - 0.5) * 2
    pxpy = torch.cat([torch.zeros(N, 1), pxpy], dim=-1)

    points = points.cuda()
    pxpy = pxpy.cuda()

    rv_images = point_to_range(points, pxpy, (3, 3))

    print(rv_images[0])

    points_reversed = range_to_point(rv_images, pxpy, mode='nearest')
    print(points_reversed)

    pass

    ## grad
    points = points.double()
    points.requires_grad = True
    check = gradcheck(point_to_range, (points, pxpy, (3, 3)))
    print(check)
