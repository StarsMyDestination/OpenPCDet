import numpy as np
from pcdet.utils.common_utils import create_logger
from pathlib import Path
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.ops.roiaware_pool3d import points_in_boxes_cpu

from pcdet.utils import visualize_utils as V
from mayavi import mlab

class_names = ['Car', 'Pedestrian', 'Cyclist']


def test_kitti_dataset():
    data_path = Path('/mnt/nas/DATA/KITTI/KITTI')
    cfg_file = '/home/jianyun/WorkSpace/pythonProjects/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset_noAug.yaml'

    cfg_from_yaml_file(cfg_file, cfg)

    logger = create_logger()
    kitti_dataset = KittiDataset(cfg, class_names=class_names, root_path=data_path, logger=logger)

    for data_dict in kitti_dataset:
        frame_id = data_dict['frame_id']
        print(f'frame_id: {frame_id}')
        points = data_dict['points']


        gt_boxes = data_dict['gt_boxes']

        point_masks = points_in_boxes_cpu(points[:, 0:3], gt_boxes[:, 0:7])

        gt_points = points[point_masks.sum(axis=0) != 0]

        fig = V.draw_scenes(points)
        V.draw_sphere_pts(gt_points, fig=fig)
        mlab.show(stop=True)


if __name__ == '__main__':
    test_kitti_dataset()
