import torch
import torch.nn as nn
import torch.nn.functional as F
from .salsanext_backbone import SalsaNextBackbone
from ...ops.rangeview import range_to_point, point_to_range


class RPBackbone(SalsaNextBackbone):
    def __init__(self, model_cfg, input_range_channels, input_point_channels):
        super().__init__(model_cfg, input_range_channels)

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_point_channels, self.cs[0]),
                nn.BatchNorm1d(self.cs[0]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(self.cs[0], self.cs[4]),
                nn.BatchNorm1d(self.cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(self.cs[4], self.cs[2]),
                nn.BatchNorm1d(self.cs[2]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(self.cs[2], self.cs[0]),
                nn.BatchNorm1d(self.cs[0]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        points = batch_dict['points'][:, 1:] # remove bs_idx
        pxpy = batch_dict['pxpy']

        x = batch_dict['spatial_features']

        # stem
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        # first RP-fusion
        p_stem = range_to_point(downCntx, pxpy)
        p_stem = p_stem + self.point_transforms[0](points)
        h, w = downCntx.shape[2:4]
        downCntx = point_to_range(p_stem, pxpy, (h, w))

        # rangeview down-stage
        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)

        # second RP-fusion
        p_stage4 = range_to_point(down3c, pxpy)
        p_stage4 = p_stage4 + self.point_transforms[1](p_stem)
        h, w = down3c.shape[2:4]
        down3c = point_to_range(p_stage4, pxpy, (h, w))

        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)

        # third RP-fusion
        p_up3e = range_to_point(up3e, pxpy)
        p_up3e = p_up3e + self.point_transforms[2](p_stage4)
        h, w = up3e.shape[2:4]
        up3e = point_to_range(p_up3e, pxpy, (h, w))

        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)

        p_up1e = range_to_point(up1e, pxpy)
        p_up1e = p_up1e + self.point_transforms[3](p_up3e)

        batch_dict['spatial_features_2d'] = up1e
        batch_dict['point_features'] = p_up1e
        return batch_dict
