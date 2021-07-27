import numpy as np
from pcdet.utils.common_utils import create_logger
from pathlib import Path
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.ops.roiaware_pool3d import points_in_boxes_cpu

from pcdet.utils import visualize_utils as V
from mayavi import mlab

class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']


def test_nus_dataset():
    data_path = Path('/mnt/nas/DATA/nuScenes/nuScenes')
    cfg_file = '/home/jianyun/WorkSpace/pythonProjects/OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset.yaml'

    cfg_from_yaml_file(cfg_file, cfg)

    logger = create_logger()
    nus_dataset = NuScenesDataset(cfg, class_names=class_names, root_path=data_path, logger=logger, training=True)
    np.random.seed(666)

    for i, data_dict in enumerate(nus_dataset):
        if i != 20:
            continue
        frame_id = data_dict['frame_id']
        print(f'frame_id: {frame_id}')
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes']

        point_masks = points_in_boxes_cpu(points[:, 0:3], gt_boxes[:, 0:7])

        gt_points = points[point_masks.sum(axis=0) != 0]

        # np.save(data_path / 'TMP' / f'gtPoints_{frame_id}.npy', gt_points)
        gt_points.astype(np.float32).tofile(data_path / 'TMP' / f'gtPoints_{frame_id}.bin')

        fig = V.draw_scenes(points)
        V.draw_sphere_pts(gt_points, fig=fig)
        mlab.show(stop=True)

        # exit(-1)


if __name__ == '__main__':
    test_nus_dataset()
