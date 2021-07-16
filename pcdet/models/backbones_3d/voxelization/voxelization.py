import torch
import torch.nn as nn
from torch.nn import functional as F
from ....ops.voxel import Voxelization


class BasicVoxelization(nn.Module):

    def __init__(self, cfg, point_cloud_range, **kwargs):
        super().__init__()
        cfg.pop('NAME')
        cfg.update({
            'point_cloud_range': point_cloud_range,
        })
        self.voxel_layer = Voxelization(**cfg)
        self.voxel_size = cfg.voxel_size
        self.grid_size = self.voxel_layer.grid_size.cpu().detach().numpy()
        self.pcd_shape = self.voxel_layer.pcd_shape

    def forward(self, batch_dict):
        """Apply hard voxelization to points."""
        points = batch_dict['points']
        bs_idx, points = points[:, 0], points[:, 1:]
        batch_size = batch_dict['batch_size']
        voxels, coors, num_points = [], [], []
        for bs_cnt in range(batch_size):
            one_point = points[bs_idx==bs_cnt]
            res_voxels, res_coors, res_num_points = self.voxel_layer(one_point)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)

        batch_dict.update({
            'voxels': voxels,
            'voxel_num_points': num_points,
            'voxel_coords': coors_batch,

        })
        return batch_dict
