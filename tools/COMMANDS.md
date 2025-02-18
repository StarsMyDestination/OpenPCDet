
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

### pointhead
sleep 300s
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv_pointhead.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36

python train.py \
--cfg_file cfgs/kitti_models/centernet_rv_pointhead.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_newP2R

### pointhead, box_coder-no_mean_size
sleep 300s
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv_pointhead.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_noMeanSize \
--set MODEL.DENSE_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.use_mean_size False



### pointhead use normal reg,
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv_pointhead.yaml \
--batch_size 4 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normReg \
--set MODEL.DENSE_HEAD.AZIMUTH_INVARIANT_REGRESSION False MODEL.DENSE_HEAD.TARGET_CONFIG.BOX_CODER PointResidualCoder


### pointhead salsanextBackbone and normal reg
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv_pointhead_salsaBack.yaml \
--batch_size 2 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normReg_salsaBackTest \
--set MODEL.DENSE_HEAD.AZIMUTH_INVARIANT_REGRESSION False MODEL.DENSE_HEAD.TARGET_CONFIG.BOX_CODER PointResidualCoder


### pointhead salsanextBackbone and normal reg 
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv_pointhead_salsaBack.yaml \
--batch_size 2 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normReg_salsaBackNewChannel \
--set MODEL.DENSE_HEAD.AZIMUTH_INVARIANT_REGRESSION False MODEL.DENSE_HEAD.TARGET_CONFIG.BOX_CODER PointResidualCoder

### pointhead salsanextBackbone and normal reg 这个版本salasanextbackbone 在up阶段用了更小的输出通道数，反而效果可能更好。
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv_pointhead_salsaBack.yaml \
--batch_size 2 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normReg_salsaBackNewChannel2 \
--set MODEL.DENSE_HEAD.AZIMUTH_INVARIANT_REGRESSION False MODEL.DENSE_HEAD.TARGET_CONFIG.BOX_CODER PointResidualCoder


===== RPBackbone =========

### pointhead RPBackbone and normal reg 
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv_pointhead_RPBackbone.yaml \
--batch_size 2 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normReg_RPBackbone \
--set MODEL.DENSE_HEAD.AZIMUTH_INVARIANT_REGRESSION False MODEL.DENSE_HEAD.TARGET_CONFIG.BOX_CODER PointResidualCoder

### pointhead RPBackbone and normal reg + iouBranchBCE
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv_pointhead_RPBackbone.yaml \
--batch_size 2 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normReg_RPBackbone_iouBCE \
--set MODEL.DENSE_HEAD.AZIMUTH_INVARIANT_REGRESSION False MODEL.DENSE_HEAD.TARGET_CONFIG.BOX_CODER PointResidualCoder


### pointhead RPBackbone and normal reg + iouBranchSmoothL1 + 
python train.py \
--cfg_file cfgs/kitti_models/centernet_rv_pointhead_RPBackbone.yaml \
--batch_size 2 --epochs 36 --workers 0 \
--extra_tag bs4_ep36_normReg_RPBackbone_iouSL1 \
--set MODEL.DENSE_HEAD.AZIMUTH_INVARIANT_REGRESSION False MODEL.DENSE_HEAD.TARGET_CONFIG.BOX_CODER PointResidualCoder


