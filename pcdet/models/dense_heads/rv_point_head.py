from functools import partial
import torch
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
from ...ops.rangeview import range_to_point
from ...utils.loss_utils import L1Loss, SmoothL1Loss


class RVPointHead(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """

    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

        self.with_iou_header = self.model_cfg.get('WITH_IOU_HEADER', False)
        if self.with_iou_header:
            self.iou_layers = self.make_fc_layers(
                fc_cfg=self.model_cfg.IOU_FC,
                input_channels=input_channels,
                output_channels=1
            )
            if self.model_cfg.IOU_LOSS_FUNC == 'SmoothL1Loss':
                self.iou_loss_func = SmoothL1Loss(reduction='sum')
            elif self.model_cfg.IOU_LOSS_FUNC == 'L1Loss':
                self.iou_loss_func = L1Loss(reduction='sum')
            elif self.model_cfg.IOU_LOSS_FUNC == 'BCE':
                self.iou_loss_func = partial(F.binary_cross_entropy, reduction='sum')
            else:
                raise NotImplementedError

        self.azimuth_invariant_reg = self.model_cfg.AZIMUTH_INVARIANT_REGRESSION

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        azimuth_rotation_matrix = input_dict['azimuth_rotation_matrix']

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True,
            azimuth_rotation_matrix=azimuth_rotation_matrix,
            ret_iou_label=self.with_iou_header, point_pred_boxes=input_dict['point_box_preds'],
        )

        return targets_dict

    def get_iou_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_iou_labels = self.forward_ret_dict['point_iou_labels']
        point_iou_preds = self.forward_ret_dict['point_iou_preds'].detach()

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_iou = self.iou_loss_func(
            point_iou_preds, point_iou_labels.detach(), weight=reg_weights
        )

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_iou = point_loss_iou * loss_weights_dict['point_iou_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_iou': point_loss_iou.item()})
        return point_loss_iou, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()
        if self.with_iou_header:
            point_loss_iou, tb_dict_3 = self.get_iou_layer_loss()

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        if self.with_iou_header:
            point_loss += point_loss_iou
            tb_dict.update(tb_dict_3)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                spatial_features_2d: range image final features, NCHW
                points: NxC, first channel is batch idx
                pxpy: Nx3, points corresponding coords in range image, [-1, 1]
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
        """
        points = batch_dict['points']
        batch_dict['point_coords'] = points[:, 0:4]

        if not 'point_features' in batch_dict:
            spatial_features_2d = batch_dict['spatial_features_2d']  # NCHW
            pxpy = batch_dict['pxpy']

            use_xyz = self.model_cfg.get('USE_XYZ', False)
            xyz = points[:, 1:4] if use_xyz else None

            point_features = range_to_point(spatial_features_2d, pxpy, use_xyz, xyz)  # (N1 + N2 + N3 + ..., C)

        else:
            point_features = batch_dict[
                'point_features']  # Fusion Backbone, like RPBackbone, already get point features

        if self.azimuth_invariant_reg is True:
            # get every points azimuth angle and rotaion angle
            theta = torch.atan2(points[:, 1], points[:, 0])
            cosa = torch.cos(theta)
            sina = torch.sin(theta)
            zeros = theta.new_zeros(points.shape[0])
            ones = theta.new_ones(points.shape[0])
            rot_matrix = torch.stack((
                cosa, -sina, zeros,
                sina, cosa, zeros,
                zeros, zeros, ones,
                theta
            ), dim=1).float()  # (N, 10)
            batch_dict['azimuth_rotation_matrix'] = rot_matrix
        else:
            batch_dict['azimuth_rotation_matrix'] = None

        ret_dict = {}
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)
        ret_dict.update({'point_cls_preds': point_cls_preds,
                         'point_box_preds': point_box_preds})
        if self.with_iou_header:
            point_iou_preds = self.iou_layers(point_features).squeeze(-1)  # (total_points,)
            point_iou_preds = torch.clamp(point_iou_preds, 0., 1.)
            ret_dict['point_iou_preds'] = point_iou_preds

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        if self.training:
            if self.with_iou_header:
                point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                    points=batch_dict['point_coords'][:, 1:4],
                    point_cls_preds=point_cls_preds, point_box_preds=point_box_preds,
                    azimuth_rotation_matrix=batch_dict['azimuth_rotation_matrix']
                )
                batch_dict['point_box_preds'] = point_box_preds.detach()

            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            if self.with_iou_header:
                ret_dict['point_iou_labels'] = targets_dict['point_iou_labels']

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds,
                azimuth_rotation_matrix=batch_dict['azimuth_rotation_matrix']
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict

        return batch_dict
