from pcdet.utils.common_utils import create_logger
from pathlib import Path
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.config import cfg, cfg_from_yaml_file

from pcdet.utils import visualize_utils as V
from mayavi import mlab

class_names = ['Car', 'Pedestrian', 'Cyclist']


def test_kitti_dataset():
    data_path = Path('/mnt/nas/DATA/KITTI/KITTI')
    cfg_file = '/home/jianyun/WorkSpace/pythonProjects/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml'

    cfg_from_yaml_file(cfg_file, cfg)

    logger = create_logger()
    kitti_dataset = KittiDataset(cfg, class_names=class_names, root_path=data_path, logger=logger)

    for data_dict in kitti_dataset:
        points = data_dict['points']
        V.draw_scenes(points)
        mlab.show(stop=True)


if __name__ == '__main__':
    test_kitti_dataset()
