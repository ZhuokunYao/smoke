import torch
from torch import nn

from .smoke_predictor import make_smoke_predictor
from .loss import make_smoke_loss_evaluator
from .post_processor import make_smoke_post_processor

from smoke.layers.utils import nms_hm

class SMOKEHead(nn.Module):
    # in_channels = 64
    def __init__(self, cfg, in_channels):
        super(SMOKEHead, self).__init__()
        
        # modify later do not modify self.cfg
        self.cfg = cfg.clone()
        # two head branch:
        # class: 6 channels , sigmoid and clamped
        # regree: 8 channels , (1, 2, 3, 2)  depth, offset, dim, ori
        #         dim is ranged into (-0.5,0.5), ori if normalized
        # return [head_class, head_regression]
        self.predictor = make_smoke_predictor(cfg, in_channels)
        
        # input: prediction, targets_train
        # output: hm_loss, reg_loss_ori + reg_loss_dim + reg_loss_loc, reg_loss_details(dict of ori & dim & loc)
        # the component of reg_loss_loc is diffirent for jdx and  waymo
        self.loss_evaluator = make_smoke_loss_evaluator(cfg)
        
        # same for jdx and waymo
        # input: prediction, targets_test
        #                                                   shifted!!       top car!!            filtered by this
        # result = torch.cat([clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores], dim=1)
        self.post_processor = make_smoke_post_processor(cfg)

    def forward(self, features, targets=None):
        x = self.predictor(features)

        if self.training:
            loss_heatmap, loss_regression, reg_loss_details = self.loss_evaluator(x, targets)
            return {}, dict(cls_loss=loss_heatmap, reg_loss=loss_regression), reg_loss_details

        if not self.training:
            x[0] = nms_hm(x[0])
            prediction = self.post_processor(x, targets)
            return prediction, {}, {}


def build_smoke_head(cfg, in_channels):
    return SMOKEHead(cfg, in_channels)
