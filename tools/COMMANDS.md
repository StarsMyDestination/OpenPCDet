
## ============ link dataset (in data dir) ============##
ln -s /mnt/nas/DATA/nuScenes/nuScenes nuscenes
ln -s /mnt/nas/DATA/KITTI/KITTI KITTI



## ============ preprocess data ============##
### run following in tools folder!
### nuScenes
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-mini

### kitti
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml


### waymo
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file cfgs/dataset_configs/waymo_dataset.yaml
    
    

## ============ training commands ============##
### nuScenes
python train.py \
--cfg_file cfgs/nuscenes_models/centernet_rv.yaml \
--batch_size 1 --epochs 20 --workers 0 \
--extra_tag baseline_ep20_obsAng_hup2_w1024



### kitti
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv.yaml \
--batch_size 8 --epochs 20 --workers 0 \
--extra_tag bs8_ep20

python train.py \
--cfg_file cfgs/kitti_models/centernet_rv.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36

python train.py \
--cfg_file cfgs/kitti_models/centernet_rv.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normInput

# ep80
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv.yaml \
--batch_size 4 --epochs 80 --workers 0 \
--extra_tag bs4_ep80_normInput

# change depth encoding
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normInput_depthNorm


python train.py \
--cfg_file cfgs/kitti_models/centernet_rv.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normInput_dirAngle \
--set MODEL.MAP_TO_RV.TRAIN_CFG.USE_OBSERVATION_ANGLE False


python train.py \
--cfg_file cfgs/kitti_models/centernet_rv.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normInput_noXYZ \
--set MODEL.MAP_TO_RV.USE_XYZ False

python train.py \
--cfg_file cfgs/kitti_models/centernet_rv.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normInput_noXYZdirAngle \
--set MODEL.MAP_TO_RV.TRAIN_CFG.USE_OBSERVATION_ANGLE False MODEL.MAP_TO_RV.USE_XYZ False

# no xyz, no angle
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normInput_noXYZ_noAngle \
--set MODEL.MAP_TO_RV.USE_XYZ False MODEL.MAP_TO_RV.USE_ANGLE False
