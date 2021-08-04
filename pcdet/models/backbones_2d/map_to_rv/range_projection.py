import numpy as np
# from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 h_fov=(-180, 180), width=2048,
                 v_fov=(-10, 30), height=32,  # if use_ringID, these two args have no effect
                 use_ringID=False, ringID_idx=-1,
                 h_upsample_ratio=1, use_xyz=True, use_angle=True,
                 ang_format='deg', norm_cfg=dict(NORM_INPUT=False)):
        self.h_fov = h_fov
        self.width = width
        self.v_fov = v_fov
        self.height = height
        self.use_ringID = use_ringID
        self.ringID_idx = ringID_idx
        self.h_upsample_ratio = h_upsample_ratio

        assert ang_format in ['deg', 'rad'], f'invalid ang_format: {ang_format}'
        if ang_format == 'deg':
            factor = 180.0 / np.pi
            self.h_fov = [x / factor for x in list(self.h_fov)]
            self.v_fov = [x / factor for x in list(self.v_fov)]

        self.use_xyz = use_xyz
        self.use_angle = use_angle

        self.norm_input = norm_cfg['NORM_INPUT']
        if self.norm_input:
            assert 'MEAN' in norm_cfg and 'STD' in norm_cfg
            self.norm_mean = torch.tensor(norm_cfg['MEAN'], dtype=torch.float32).cuda()
            self.norm_std = torch.tensor(norm_cfg['STD'], dtype=torch.float32).cuda()
        else:
            self.norm_mean, self.norm_std = None, None

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

    def points_to_rvImage(self, points):
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

        w = self.width
        h = self.height
        # if self.use_ringID:
        #     assert max(points[:, self.ringID_idx]) < h

        u = (uv_normed[:, 0] * w).long()

        if not self.use_ringID:
            features = points[:, 3:]
            v = (uv_normed[:, 1] * h).long()
        else:
            ringID = points[:, self.ringID_idx]
            fea_chan_list = list(range(points.shape[1]))
            fea_chan_list.pop(self.ringID_idx)
            features = points[:, fea_chan_list[3:]]  # remove ringID in features
            v = ringID.long()

        # get pxpy
        px = (uv_normed[:, 0] - 0.5) * 2  # [-1, 1]
        py = (v / h - 0.5) * 2  # [-1, 1]
        pxpy = torch.stack([px, py], dim=-1)

        # gather features
        if self.use_xyz and self.use_angle:
            features = torch.cat([rThetaPhi, points[:, 0:3], features], dim=1)  # Nx(6+C)
            if self.norm_input:
                features[:, 0:7] = (features[:, 0:7] - self.norm_mean) / self.norm_std  # rThetaPhi xyz intensity
        elif self.use_xyz and (not self.use_angle):
            features = torch.cat([rThetaPhi[:, 0:1], points[:, 0:3], features], dim=1)  # Nx(4+C)
            if self.norm_input:
                features[:, 0:5] = (features[:, 0:5] - self.norm_mean[[0, 3, 4, 5, 6]]) / \
                                   self.norm_std[[0, 3, 4, 5, 6]]  # r xyz intensity
        elif (not self.use_xyz) and self.use_angle:
            features = torch.cat([rThetaPhi, features], dim=1)  # Nx(3+C)
            if self.norm_input:
                features[:, 0:4] = (features[:, 0:4] - self.norm_mean[[0, 1, 2, 6]]) / \
                                   self.norm_std[[0, 1, 2, 6]]  # rThetaPhi intensity
        elif (not self.use_xyz) and (not self.use_angle):
            features = torch.cat([rThetaPhi[:, 0:1], features], dim=1)  # Nx(1+C)
            if self.norm_input:
                features[:, 0:2] = (features[:, 0:2] - self.norm_mean[[0, 6]]) / \
                                   self.norm_std[[0, 6]]  # r intensity

        # init range_image
        channels = features.shape[1]
        rv_image = points.new_zeros((h, w, channels))

        rv_image[v, u] = features

        rv_image = rv_image.permute(2, 0, 1).unsqueeze(0).contiguous()  # 1CHW

        if self.h_upsample_ratio != 1:
            dst_size = (int(h * self.h_upsample_ratio), w)
            rv_image = F.interpolate(rv_image, dst_size, mode='bilinear')

        return rv_image, pxpy


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
        self.use_angle = cfg.get('USE_ANGLE', True)  # use theta, phi as features

        if self.use_ringID:
            self.num_rv_features -= 1  # remove rangID in features

        if self.use_xyz:
            self.num_rv_features += 3  # add xyz in features

        if not self.use_angle:
            self.num_rv_features -= 2

        proj_cfg = {
            'h_fov': cfg.H_FOV,
            'width': cfg.WIDTH,
            'v_fov': cfg.V_FOV,
            'height': cfg.HEIGHT,
            'use_ringID': self.use_ringID,
            'ringID_idx': cfg.RINGID_IDX,
            'ang_format': cfg.get('ANG_FORMAT', 'deg'),
            'use_xyz': self.use_xyz,
            'use_angle': self.use_angle,
            'h_upsample_ratio': cfg.get('H_UPSAMPLE_RATIO', 1),
            'norm_cfg': cfg.get('NORM_CFG', dict(NORM_INPUT=False)),
        }

        self.rp_trans_api = RPTransformation(**proj_cfg)

        self.filter_gt_boxes = cfg.TRAIN_CFG.FILTER_GT_BOXES
        self.use_observation_angle = cfg.TRAIN_CFG.USE_OBSERVATION_ANGLE

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
            rv_image, pxpy = self.rp_trans_api.points_to_rvImage(one_point)  # 1CHW
            rv_images.append(rv_image)
        rv_images = torch.cat(rv_images, dim=0)  # NCHW

        batch_dict.update({
            'spatial_features': rv_images,
            'rv_img_shape': rv_images.shape[2:4],
            'pxpy': pxpy,
        })

        if self.training:
            gt_boxes = batch_dict['gt_boxes']
            gt_boxes_rv = gt_boxes.new_zeros((batch_size, gt_boxes.shape[1], 4))  # [cx, cy, w, h]
            num_valid_gt = batch_dict['num_valid_gt']
            for bs_cnt in range(batch_size):
                one_n_valid_gt = num_valid_gt[bs_cnt]
                one_gt_box = gt_boxes[bs_cnt, :one_n_valid_gt].clone()
                center = one_gt_box[:, 0:3]
                center_uv_normed = self.rp_trans_api.xyz_to_uvNormed(center)

                if self.filter_gt_boxes:
                    mask = (center_uv_normed[:, 0] >= 0) & (center_uv_normed[:, 0] < 1) & \
                           (center_uv_normed[:, 1] >= 0) & (center_uv_normed[:, 1] < 1)
                    one_gt_box = one_gt_box[mask]
                    center_uv_normed = center_uv_normed[mask]
                    one_n_valid_gt = mask.sum()
                    gt_boxes[bs_cnt] = 0
                    gt_boxes[bs_cnt, :one_n_valid_gt] = one_gt_box
                else:
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

                if self.use_observation_angle:
                    theta = torch.atan2(one_gt_box[:, 1], one_gt_box[:, 0])
                    one_gt_box[:, 6] -= theta
                    gt_boxes[bs_cnt, :one_n_valid_gt] = one_gt_box

            batch_dict['gt_boxes_rv'] = gt_boxes_rv  # normed [cx cy w h]
        return batch_dict
