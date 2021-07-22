import numpy as np
from functools import partial

import torch
import torch.nn as nn

from ....utils.box_utils import boxes_to_corners_3d


class SphereProjection(object):
    def __init__(self):
        '''
        Cartesian: x-front, y-left, z-up
        Sphere: r-range; theta-azimuth, angle with +x, clockwise; phi-elevation, angle with xy plane, clockwise
        note that, such sphere def. is convenient for vis.
        '''
        pass

    @classmethod
    def xyz_to_rThetaPhi(cls, xyz):
        '''

        :param xyz: Nx3
        :return: r: range, theta: azimuth, phi: elevation
        '''
        r = torch.linalg.norm(xyz, ord=2, dim=1, keepdim=True)  # Nx1
        theta = -torch.atan2(xyz[:, 1], xyz[:, 0]).unsqueeze(-1)  # Nx1
        phi = -torch.asin(xyz[:, 2] / torch.clamp_min(r.squeeze(), 1e-5)).unsqueeze(-1)  # Nx1
        return torch.cat([r, theta, phi], dim=-1)  # Nx3

    @classmethod
    def rThetaPhi_to_xyz(cls, rThetaPhi):
        r, theta, phi = torch.split(rThetaPhi, 1, dim=-1)
        z = -r * torch.sin(phi)
        xy = r * torch.cos(phi)
        x = xy * torch.cos(theta)
        y = -xy * torch.sin(theta)
        return torch.cat([x, y, z], dim=-1)  # Nx3


class RPTransformation(object):
    '''
    trans between Range and Point.
    '''

    def __init__(self,
                 h_fov=(-180, 180), h_res=0.02,
                 v_fov=(-10, 30), v_res=1.2,  # if use_ringID, these two args have no effect
                 use_ringID=False, ringID_idx=-1,
                 **kwargs):
        self.h_fov = h_fov
        self.h_res = h_res
        self.v_fov = v_fov
        self.v_res = v_res
        self.use_ringID = use_ringID
        self.ringID_idx = ringID_idx

        ang_format = kwargs.get('ang_format', 'deg')  # 'deg' or 'rad'
        assert ang_format in ['deg', 'rad'], f'invalid ang_format: {ang_format}'
        if ang_format == 'deg':
            factor = 180.0 / np.pi
            self.h_fov = [x / factor for x in list(self.h_fov)]
            self.h_res /= factor
            self.v_fov = [x / factor for x in list(self.v_fov)]
            self.v_res /= factor

        self.use_xyz = kwargs.get('use_xyz', True)

    def xyz_to_uvNormed(self, xyz, return_more=False):
        '''
        :param uv_normed: Nx2, coords in rv_image, normed to 0~1
        :param range: Nx1, range
        :return: Nx3
        '''
        rThetaPhi = SphereProjection.xyz_to_rThetaPhi(xyz)
        u_normed = (rThetaPhi[:, 1:2] - self.h_fov[0]) / (self.h_fov[1] - self.h_fov[0])
        v_normed = (rThetaPhi[:, 2:3] - self.v_fov[0]) / (self.v_fov[1] - self.v_fov[0])

        uv_normed = torch.cat([u_normed, v_normed], dim=1)  # Nx2
        if return_more:
            return uv_normed, rThetaPhi
        else:
            return uv_normed

    def uvNormed_to_xyz(self, uv_normed, range):
        '''
        :param uv_normed: Nx2, coords in rv_image, normed to 0~1
        :param range: Nx1, range
        :return: Nx3
        '''
        assert not self.use_ringID, 'if use ringID as range image height, then it is not compatible to convert uv back to xyz!'
        assert len(range.shape) == 2 and range.shape[1] == 1
        u, v = torch.split(uv_normed, 1, dim=-1)  # Nx1, Nx1
        theta = u * (self.h_fov[1] - self.h_fov[0]) + self.h_fov[0]
        phi = v * (self.v_fov[1] - self.v_fov[0]) + self.v_fov[0]

        rThetaPhi = torch.cat([range, theta, phi], dim=-1)

        return SphereProjection.rThetaPhi_to_xyz(rThetaPhi)  # Nx3

    def points_to_rvImage(self, points, ):
        xyz = points[:, 0:3]
        uv_normed, rThetaPhi = self.xyz_to_uvNormed(xyz, return_more=True)

        # mask outside points
        mask1 = (uv_normed[:, 0] >= 0) & (uv_normed[:, 0] < 1)
        if not self.use_ringID:
            mask2 = (uv_normed[:, 1] >= 0) & (uv_normed[:, 1] < 1)
            mask = mask1 & mask2
        else:
            mask = mask1
        points = points[mask]
        uv_normed = uv_normed[mask]
        rThetaPhi = rThetaPhi[mask]

        w = int((self.h_fov[1] - self.h_fov[0]) / self.h_res)
        h = max(points[:, self.ringID_idx]) if self.use_ringID else int((self.v_fov[1] - self.v_fov[0]) / self.v_res)

        u = (uv_normed[:, 0] * w).long()

        if not self.use_ringID:
            features = points[:, 3:]
            v = (uv_normed[:, 1] * h).long()
        else:
            ringID = points[:, self.ringID_idx]
            features = points[:, list(range(points.shape[1])).pop(self.ringID_idx)[3:]]  # remove ringID in features
            v = ringID.long()

        # gather features
        if self.use_xyz:
            features = torch.cat([points[:, 0:3], rThetaPhi, features], dim=1)  # Nx(6+C)
        else:
            features = torch.cat([rThetaPhi, features], dim=1)  # Nx(3+C)

        # init range_image
        channels = features.shape[1]
        rv_image = points.new_zeros((h, w, channels))

        rv_image[v, u] = features

        return rv_image.permute(2, 0, 1).contiguous()


class BasicRangeProjection(nn.Module):
    def __init__(self, cfg, input_channels, **kwargs):
        '''
        method to transfer point cloud to range view.
        1. use ringID as range image rows: can partition space non-uniformly(some lidar vertical resolution is non-uniform),
           and keeps most information. wheres, it's not easy to restore original Cartesian coordinates.
        2. use spherical projection for rows too: may lose some raw information, but can restore original Cartesian coordinates easily.
        '''

        super().__init__()
        self.cfg = cfg
        self.num_rv_features = input_channels

        self.use_ringID = cfg.USE_RINGID
        self.use_xyz = cfg.get('USE_XYZ', True)

        if self.use_ringID:
            self.num_rv_features -= 1  # remove rangID in features

        if self.use_xyz:
            self.num_rv_features += 3  # add xyz in features

        proj_cfg = {
            'h_fov': cfg.H_FOV,
            'h_res': cfg.H_RES,
            'v_fov': cfg.V_FOV,
            'v_res': cfg.V_RES,
            'use_ringID': self.use_ringID,
            'ringID_idx': cfg.RINGID_IDX,
            'ang_format': cfg.get('ANG_FORMAT', 'deg'),
            'use_xyz': self.use_xyz,
        }

        self.rp_trans_api = RPTransformation(**proj_cfg)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                points: stacked point cloud, first column is batch_idx
        Returns:
            batch_dict:
                spatial_features:

        """

        points = batch_dict['points']
        bs_idx, points = points[:, 0], points[:, 1:]
        batch_size = batch_dict['batch_size']

        rv_images = []
        for bs_cnt in range(batch_size):
            one_point = points[bs_idx == bs_cnt]
            rv_image = self.rp_trans_api.points_to_rvImage(one_point).unsqueeze(0)
            rv_images.append(rv_image)
        rv_images = torch.cat(rv_images, dim=0)  # NCHW

        batch_dict.update({
            'spatial_features': rv_images,
            'rv_img_shape': rv_images.shape[2:4]
        })

        if self.training:
            # TODO maybe filter objects outside here.
            gt_boxes = batch_dict['gt_boxes']
            gt_boxes_rv = gt_boxes.new_zeros((batch_size, gt_boxes.shape[1], 4))  # [cx, cy, w, h]
            num_valid_gt = batch_dict['num_valid_gt']
            for bs_cnt in range(batch_size):
                one_n_valid_gt = num_valid_gt[bs_cnt]
                one_gt_box = gt_boxes[bs_cnt, :one_n_valid_gt]
                center = one_gt_box[:, 0:3]
                center_uv_normed = self.rp_trans_api.xyz_to_uvNormed(center)
                center_uv_normed = torch.clamp(center_uv_normed, 0, 1 - 1e-3)
                ##!! gives wrong answer when using torch.matmul in 'rotate_points_along_z' on cuda
                # due to pip-installed pytorch1.8.1+cu111 bug(https://github.com/pytorch/pytorch/issues/56747)!!
                # corners = boxes_to_corners_3d(one_gt_box.cpu()).to(one_gt_box.device)  # Nx8x3
                ## !! use conda installed pytorch !!
                corners = boxes_to_corners_3d(one_gt_box)  # Nx8x3
                corners_uv_normed = self.rp_trans_api.xyz_to_uvNormed(corners.view(-1, 3)).view(-1, 8, 2)
                corners_uv_min = torch.clamp_min(torch.min(corners_uv_normed, dim=1)[0], 0)
                corners_uv_max = torch.clamp_max(torch.max(corners_uv_normed, dim=1)[0], 1)
                wh_normed = corners_uv_max - corners_uv_min
                gt_boxes_rv[bs_cnt, :one_n_valid_gt] = torch.cat([center_uv_normed, wh_normed], dim=1)

            batch_dict['gt_boxes_rv'] = gt_boxes_rv  # normed [cx cy w h]
        return batch_dict
