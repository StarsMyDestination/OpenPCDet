import glob
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from easydict import EasyDict

from pcdet.models.backbones_2d.map_to_rv.range_projection import SphereProjection, BasicRangeProjection
from pcdet.utils import visualize_utils as V
from mayavi import mlab
import cmapy


def test_sphere_projection():
    xyz = torch.rand(4, 3)
    print(f'x: \n {xyz}')
    rThetaPhi = SphereProjection.xyz_to_rThetaPhi(xyz)
    print(f'rThetaPhi: \n {rThetaPhi}')

    new_xyz = SphereProjection.rThetaPhi_to_xyz(rThetaPhi)
    print(f'new_xyz: \n {new_xyz}')

    assert torch.abs(xyz - new_xyz).sum() < 1e-5


def _norm_0_255(image):
    shape = image.shape
    image = image.reshape(-1, shape[-1])
    mask = (image != 0)
    c_min = image[mask].min(0)
    c_max = image.max(0)
    image = (image - c_min) / (c_max - c_min) * 255.0
    image *= mask
    return image.reshape(*shape)


def _draw_quarter_line(image):
    h, w, c = image.shape
    ws = [int(w * x) for x in [0.25, 0.5, 0.75]]
    image[:, ws, :] = 255
    return image


def test_points_to_rvImage():
    ### nuScenes
    # data_folder = r'/mnt/nas/DATA/nuScenes/nuScenes/samples/LIDAR_TOP'
    # data_folder = r'/mnt/nas/DATA/nuScenes/nuScenes/TMP'
    # lidar_channels = 5

    ### KITTI
    data_folder = r'/mnt/nas/DATA/KITTI/KITTI/training/velodyne'
    lidar_channels = 4

    data_file_list = sorted(glob.glob(str(data_folder + '/*bin')))
    # index = np.random.choice(len(data_file_list), 1)[0]
    index = 0
    print(data_file_list[index])
    points = np.fromfile(data_file_list[index], dtype=np.float32).reshape(-1, lidar_channels)

    points = torch.from_numpy(points)
    points = F.pad(points, (1, 0), mode='constant', value=0)

    cfg = EasyDict()

    ### nuScenes
    cfg.USE_RINGID = False
    cfg.RINGID_IDX = -1
    cfg.USE_XYZ = True
    cfg.H_FOV = [-180, 180]
    cfg.WIDTH = 1024
    # cfg.H_FOV = [-45, 45]
    # cfg.WIDTH = 512
    cfg.V_FOV = [-10, 30]
    cfg.HEIGHT = 32
    cfg.H_UPSAMPLE_RATIO = 1

    ### KITTI
    cfg.USE_RINGID = False
    cfg.RINGID_IDX = -1
    cfg.USE_XYZ = True
    cfg.H_FOV = [-41, 41]
    cfg.WIDTH = 512
    cfg.V_FOV = [-5, 15]
    cfg.HEIGHT = 64
    cfg.H_UPSAMPLE_RATIO = 1

    cfg.TRAIN_CFG = EasyDict()
    cfg.TRAIN_CFG.FILTER_GT_BOXES = True
    cfg.TRAIN_CFG.USE_OBSERVATION_ANGLE = True

    rp = BasicRangeProjection(cfg, input_channels=5)
    rp.eval()

    ret = rp(dict(points=points, batch_size=1))

    # draw depth image
    rv_image = ret['spatial_features'][0].permute(1, 2, 0).numpy()  # rThetaPhi, xyz, ...
    depth = rv_image[:, :, 0:1]
    depth[depth > 0] = depth[depth > 0] ** (1 / 4)
    depth = _norm_0_255(depth).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth, cmapy.cmap('viridis'))
    # draw line
    depth_colored = _draw_quarter_line(depth_colored)
    # cv2.imwrite('nus_rv32.png', depth_colored)
    cv2.imwrite('kitti_rv64.png', depth_colored)

    V.draw_scenes(points[:, 1:])
    mlab.show(stop=True)


if __name__ == '__main__':
    test_sphere_projection()

    test_points_to_rvImage()
