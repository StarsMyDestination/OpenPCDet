import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file, cfg_from_list
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--extra_tag', type=str, default='debug')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    ## TODO DEBUG
    if args.extra_tag == 'debug':
        ## nuScenes
        # args.cfg_file = 'cfgs/nuscenes_models/centernet_rv.yaml'
        # args.ckpt = '../output/nuscenes_models/centernet_rv/baseline_ep20_obsAng_hup2_w1024/ckpt/checkpoint_epoch_20.pth'

        ## KITTI
        # args.cfg_file = 'cfgs/kitti_models/centernet_rv.yaml'
        # args.ckpt = '../output/kitti_models/centernet_rv/bs4_ep36_normInput/ckpt/checkpoint_epoch_36.pth'

        args.cfg_file = 'cfgs/kitti_models/centernet_rv_pointhead.yaml'
        args.ckpt = '../output/kitti_models/centernet_rv_pointhead/bs4_ep36/ckpt/checkpoint_epoch_36.pth'

        # normal regression
        # args.ckpt = '../output/kitti_models/centernet_rv_pointhead/bs4_ep36_normReg/ckpt/checkpoint_epoch_36.pth'
        # args.set_cfgs = ['MODEL.DENSE_HEAD.AZIMUTH_INVARIANT_REGRESSION', False,
        #                  'MODEL.DENSE_HEAD.TARGET_CONFIG.BOX_CODER', 'PointResidualCoder']

    ## END

    cfg_from_yaml_file(args.cfg_file, cfg)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=0, logger=logger, training=False
    )
    logger.info(f'Total number of samples: \t{len(test_set)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(test_loader):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0],
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
