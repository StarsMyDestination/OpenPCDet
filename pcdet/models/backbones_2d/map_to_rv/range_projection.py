import numpy as np
from functools import partial

import torch
import torch.nn as nn


class SphereProjection(object):
    def __init__(self):
        '''
        Cartesian: x-front, y-left, z-up
        Sphere: r-range; theta-azimuth, angle with +x, clockwise; phi-elevation, angle with xy plane, clockwise
        note that, such sphere def. is convenient for vis.
        '''
        pass

    @classmethod
    def xyz_to_rThetaPhi(cls, xyz, compute_phi=True):
        '''

        :param xyz: Nx3
        :return: r: range, theta: azimuth, phi: elevation
        '''
        r = torch.linalg.norm(xyz, ord=2, dim=1, keepdim=True)  # Nx1
        theta = -torch.atan2(xyz[:, 1], xyz[:, 0]).unsqueeze(-1)  # Nx1

        if compute_phi:
            phi = -torch.asin(xyz[:, 2] / torch.clamp_min(r.squeeze(), 1e-5)).unsqueeze(-1)  # Nx1
            return torch.cat([r, theta, phi], dim=-1)  # Nx3
        else:
            return torch.cat([r, theta], dim=-1)  # Nx2

    @classmethod
    def rThetaPhi_to_xyz(cls, rThetaPhi):
        r, theta, phi = torch.split(rThetaPhi, 1, dim=-1)
        z = -r * torch.sin(phi).unsqueeze(-1)
        xy = r * torch.cos(phi)
        x = xy * torch.cos(theta).unsqueeze(-1)
        y = -xy * torch.sin(theta).unsqueeze(-1)
        return torch.cat([x, y, z], dim=-1)  # Nx3


def points_to_rvImage(points,
                      h_fov=(-180, 180), h_res=0.02,
                      v_fov=(-10, 30), v_res=1.2,  # if use_ringID, these two args have no effect
                      use_ringID=False, ringID_idx=-1,
                      **kwargs):
    ang_format = kwargs.get('ang_format', 'deg')  # 'deg' or 'rad'
    assert ang_format in ['deg', 'rad'], f'invalid ang_format: {ang_format}'
    if ang_format == 'deg':
        factor = 180.0 / np.pi
        h_fov = [x / factor for x in list(h_fov)]
        h_res /= factor
        v_fov = [x / factor for x in list(v_fov)]
        v_res /= factor

    xyz = points[:, 0:3]
    rThetaPhi = SphereProjection.xyz_to_rThetaPhi(xyz)

    # mask outside points
    mask1 = (rThetaPhi[:, 1] >= h_fov[0]) & (rThetaPhi[:, 1] < h_fov[1])
    if not use_ringID:
        mask2 = (rThetaPhi[:, 2] >= v_fov[0]) & (rThetaPhi[:, 2] < v_fov[1])
        mask = mask1 & mask2
    else:
        mask = mask1
    points = points[mask]
    rThetaPhi = rThetaPhi[mask]

    u = ((rThetaPhi[:, 1] - h_fov[0]) / h_res).long()

    if not use_ringID:
        features = points[:, 3:]
        v = ((rThetaPhi[:, 2] - v_fov[0]) / v_res).long()
    else:
        ringID = points[:, 0:3], points[:, ringID_idx]
        features = points[:, list(range(points.shape[1])).pop(ringID_idx)[3:]]
        v = ringID.long()

    # gather features
    use_xyz = kwargs.get('use_xyz', True)
    if use_xyz:
        features = torch.cat([points[:, 0:3], rThetaPhi, features], dim=1)  # Nx(6+C)
    else:
        features = torch.cat([rThetaPhi, features], dim=1)  # Nx(3+C)

    # init range_image
    w = int((h_fov[1] - h_fov[0]) / h_res)
    h = max(ringID) if use_ringID else int((v_fov[1] - v_fov[0]) / v_res)
    channels = features.shape[1]
    rv_image = points.new_zeros((h, w, channels))

    rv_image[v, u] = features

    return rv_image.permute(2, 0, 1).contiguous()


def rvImage_to_points(rv_image,
                      h_fov=(-180, 180), h_res=0.02,
                      v_fov=(-10, 30), v_res=1.2,  # if use_ringID, these two args have no effect
                      use_ringID=False, ringID_idx=-1,
                      **kwargs):
    raise NotImplementedError  # TODO if needed?


def uv_to_xyz(uv_normed, range,
              h_fov=(-180, 180), v_fov=(-10, 30),
              **kwargs):
    '''
    :param uv_normed: Nx2, coords in rv_image, normed to 0~1
    :param range: Nx1, range
    :return: Nx3
    '''
    ang_format = kwargs.get('ang_format', 'deg')  # 'deg' or 'rad'
    assert ang_format in ['deg', 'rad'], f'invalid ang_format: {ang_format}'
    if ang_format == 'deg':
        factor = 180.0 / np.pi
        h_fov = (x / factor for x in list(h_fov))
        v_fov = (x / factor for x in list(v_fov))

    u, v = torch.split(uv_normed, 1, dim=-1)
    theta = u.unsqueeze(-1) * (h_fov[1] - h_fov[0]) + h_fov[0]
    phi = v.unsqueeze(-1) * (v_fov[1] - v_fov[0]) + v_fov[0]

    rThetaPhi = torch.cat([range, theta, phi], dim=-1)

    return SphereProjection.rThetaPhi_to_xyz(rThetaPhi)


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

        self.proj_func = partial(points_to_rvImage, **proj_cfg)

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
            rv_image = self.proj_func(one_point).unsqueeze(0)
            rv_images.append(rv_image)
        rv_images = torch.cat(rv_images, dim=0)  # NCHW

        batch_dict.update({
            'spatial_features': rv_images,
        })
        return batch_dict
