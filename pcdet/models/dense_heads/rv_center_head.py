import torch
import torch.nn as nn
from ...utils.common_utils import bias_init_with_prob, normal_init, multi_apply
# from mmcv.ops import batched_nms

from ...utils.loss_utils import GaussianFocalLoss, L1Loss
from ...utils.gaussian_target import (gaussian_radius, gen_gaussian_target,
                                      get_local_maximum, get_topk_from_heatmap,
                                      transpose_and_gather_feat)


class RVCenterNetHead(nn.Module):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, model_cfg, input_channels, num_class, rp_trans_api, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_classes = num_class
        self.rp_trans_api = rp_trans_api

        feat_channel = model_cfg.FEAT_CHANNEL
        self.heatmap_head = self._build_head(input_channels, feat_channel, num_class)
        self.lwh_head = self._build_head(input_channels, feat_channel, 3)
        self.offset_head = self._build_head(input_channels, feat_channel, 2)
        self.depth_head = self._build_head(input_channels, feat_channel, 1)
        self.dir_head = self._build_head(input_channels, feat_channel, 2)

        loss_weights = model_cfg.LOSS_CONFIG.LOSS_WEIGHTS

        self.loss_center_heatmap = GaussianFocalLoss(loss_weight=loss_weights['heatmap_cls_weight'])
        self.loss_lwh = L1Loss(loss_weight=loss_weights['lwh_reg_weight'])
        self.loss_offset = L1Loss(loss_weight=loss_weights['offset_reg_weight'])
        self.loss_depth = L1Loss(loss_weight=loss_weights['depth_reg_weight'])
        self.loss_dir = L1Loss(loss_weight=loss_weights['dir_reg_weight'])

        self.forward_ret_dict = {}

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.lwh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, data_dict):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        spatial_features_2d = data_dict['spatial_features_2d']
        if not isinstance(spatial_features_2d, list):
            spatial_features_2d = [spatial_features_2d]
        center_heatmap_preds, lwh_preds, offset_preds, depth_preds, dir_preds = \
            multi_apply(self.forward_single, spatial_features_2d)

        self.forward_ret_dict['center_heatmap_preds'] = center_heatmap_preds  # list
        self.forward_ret_dict['lwh_preds'] = lwh_preds  # list
        self.forward_ret_dict['offset_preds'] = offset_preds  # list
        self.forward_ret_dict['depth_preds'] = depth_preds  # list
        self.forward_ret_dict['dir_preds'] = dir_preds  # list

        if self.training:
            self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']
            self.forward_ret_dict['gt_boxes_rv'] = data_dict['gt_boxes_rv']
            self.forward_ret_dict['num_valid_gt'] = data_dict['num_valid_gt']
            # self.forward_ret_dict['rv_img_shape'] = data_dict['rv_img_shape']

        else:
            batch_box_preds, batch_cls_labels, batch_cls_scores \
                = self.generate_predicted_boxes(center_heatmap_preds, lwh_preds,
                                                offset_preds, depth_preds,
                                                dir_preds, self.rp_trans_api)

            data_dict['batch_cls_labels'] = batch_cls_labels
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_cls_scores'] = batch_cls_scores
        return data_dict

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        lwh_pred = self.lwh_head(feat)
        offset_pred = self.offset_head(feat)
        depth_pred = self.depth_head(feat)
        dir_pred = self.dir_head(feat)
        return center_heatmap_pred, lwh_pred, offset_pred, depth_pred, dir_pred

    def get_loss(self, tb_dict=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        tb_dict = {} if tb_dict is None else tb_dict
        center_heatmap_preds = self.forward_ret_dict['center_heatmap_preds']
        lwh_preds = self.forward_ret_dict['lwh_preds']
        offset_preds = self.forward_ret_dict['offset_preds']
        depth_preds = self.forward_ret_dict['depth_preds']
        dir_preds = self.forward_ret_dict['dir_preds']

        gt_boxes = self.forward_ret_dict['gt_boxes']
        gt_boxes_rv = self.forward_ret_dict['gt_boxes_rv']
        n_valid_gt = self.forward_ret_dict['num_valid_gt']
        # rv_img_shape = self.forward_ret_dict['rv_img_shape']

        assert len(center_heatmap_preds) == len(lwh_preds) == len(offset_preds) \
               == len(depth_preds) == len(dir_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        lwh_pred = lwh_preds[0]
        offset_pred = offset_preds[0]
        depth_pred = depth_preds[0]
        dir_pred = dir_preds[0]

        target_result, avg_factor = self.get_targets(gt_boxes, gt_boxes_rv, n_valid_gt,
                                                     center_heatmap_pred.shape)

        center_heatmap_target = target_result['center_heatmap_target']
        lwh_target = target_result['lwh_target']
        offset_target = target_result['offset_target']
        depth_target = target_result['depth_target']
        dir_target = target_result['dir_target']
        reg_weight = target_result['reg_weight']

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target,
                                                       avg_factor=avg_factor)
        loss_lwh = self.loss_lwh(lwh_pred, lwh_target, reg_weight,
                                 avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(offset_pred, offset_target, reg_weight,
                                       avg_factor=avg_factor * 2)
        loss_depth = self.loss_depth(depth_pred, depth_target, reg_weight,
                                     avg_factor=avg_factor * 2)
        loss_dir = self.loss_dir(dir_pred, dir_target, reg_weight,
                                 avg_factor=avg_factor * 2)

        loss = loss_center_heatmap + loss_lwh + loss_depth + loss_dir

        tb_dict.update({
            'loss_center_heatmap': loss_center_heatmap,
            'loss_lwh': loss_lwh,
            'loss_offset': loss_offset,
            'loss_depth': loss_depth,
            'loss_dir': loss_dir,
        })
        return loss, tb_dict

    def get_targets(self, gt_boxes, gt_boxes_rv, n_valid_gt, feat_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_boxes (Tensor): Ground truth 3d bboxes with
                shape (B, num_gts, 7) in [x, y, z, l, w, h, heading, (vx, vy,), cls] format.
            gt_boxes_rv (Tensor): normed Ground truth bboxes for each range view image with
                shape (B, num_gts, 4) in normed [u, v, w, h] format.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - lwh_target (Tensor): targets of lwh predict, shape \
                   (B, 3, H, W).
               - lwh_target (Tensor): targets of lwh predict, shape \
                   (B, 3, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - depth_target (Tensor): targets of depth predict, shape \
                   (B, 1, H, W).
               - dir_target (Tensor): targets of direction predict, shape \
                   (B, 2, H, W).
               - reg_weight (Tensor): weights of regression target predict, shape \
                   (B, 2, H, W).
        """
        bs, _, feat_h, feat_w = feat_shape

        center_heatmap_target = gt_boxes.new_zeros([bs, self.num_classes, feat_h, feat_w])
        lwh_target = gt_boxes.new_zeros([bs, 3, feat_h, feat_w])
        offset_target = gt_boxes.new_zeros([bs, 2, feat_h, feat_w])
        depth_target = gt_boxes.new_zeros([bs, 1, feat_h, feat_w])
        dir_target = gt_boxes.new_zeros([bs, 2, feat_h, feat_w])
        reg_weight = gt_boxes[-1].new_zeros([bs, 1, feat_h, feat_w])

        for batch_id in range(bs):
            gt_box = gt_boxes[batch_id][:n_valid_gt[batch_id]]
            gt_box, gt_label = gt_box[:, 0:-1], gt_box[:, -1].long()
            gt_box_rv = gt_boxes_rv[batch_id][:n_valid_gt[batch_id]]

            gt_centers = gt_box_rv[:, 0:2]
            gt_centers[:, 0] *= feat_w
            gt_centers[:, 1] *= feat_h

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = gt_box_rv[j][3] * feat_h
                scale_box_w = gt_box_rv[j][2] * feat_w
                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j] - 1
                gen_gaussian_target(center_heatmap_target[batch_id, ind], [ctx_int, cty_int], radius)

                lwh_target[batch_id, :, cty_int, ctx_int] = gt_box[j, 3:6]

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                depth_target[batch_id, :, cty_int, ctx_int] = torch.log(torch.norm(gt_box[j, 0:3]) + 1)

                dir_target[batch_id, 0, cty_int, ctx_int] = torch.cos(gt_box[j, 6])
                dir_target[batch_id, 1, cty_int, ctx_int] = torch.sin(gt_box[j, 6])

                reg_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            lwh_target=lwh_target,
            offset_target=offset_target,
            depth_target=depth_target,
            dir_target=dir_target,
            reg_weight=reg_weight)
        return target_result, avg_factor

    def generate_predicted_boxes(self,
                                 center_heatmap_preds,
                                 lwh_preds,
                                 offset_preds,
                                 depth_preds,
                                 dir_preds,
                                 rp_trans_api):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            rp_trans_api (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(lwh_preds) == len(offset_preds) \
               == len(depth_preds) == len(dir_preds) == 1

        batch_det_bboxes, batch_labels, batch_scores = self.decode_heatmap(
            center_heatmap_preds[0],
            lwh_preds[0],
            offset_preds[0],
            depth_preds[0],
            dir_preds[0],
            rp_trans_api,
            k=self.model_cfg.TEST_CFG.TOP_K,
            kernel=self.model_cfg.TEST_CFG.LOCAL_MAXIMUM_KERNEL)

        return batch_det_bboxes, batch_labels, batch_scores

    def decode_heatmap(self,
                       center_heatmap_pred,
                       lwh_pred,
                       offset_pred,
                       depth_pred,
                       dir_pred,
                       rp_trans_api,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            lwh_pred (Tensor): wh predict, shape (B, 3, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            depth_pred (Tensor): depth predict, shape (B, 1, H, W).
            dir_pred (Tensor): direction predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]

        center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        lwh = transpose_and_gather_feat(lwh_pred, batch_index)  # (B, topK, 3)
        offset = transpose_and_gather_feat(offset_pred, batch_index)  # (B, topK, 2)
        depth = transpose_and_gather_feat(depth_pred, batch_index)  # (B, topK, 1)
        dir = transpose_and_gather_feat(dir_pred, batch_index)  # (B, topK, 2)

        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        u_normed = topk_xs / width
        v_normed = topk_ys / height
        uv_normed = torch.stack([u_normed, v_normed], dim=2)

        depth = torch.exp(depth) - 1
        center_xyz = []
        for one_uv_normed, one_depth in zip(uv_normed, depth):
            center_xyz.append(rp_trans_api.uvNormed_to_xyz(one_uv_normed, one_depth))
        center_xyz = torch.stack(center_xyz, dim=0)

        dir = torch.atan2(dir[..., 1], dir[..., 0])

        batch_bboxes = torch.cat([center_xyz, lwh, dir[..., None]], dim=-1)

        return batch_bboxes, batch_topk_labels, batch_scores
