import torch
from torch import nn
from torch.nn import functional as F

from smoke.modeling.smoke_coder import SMOKECoder
from smoke.modeling.heads.smoke_head.smoke_predictor import get_channel_spec
from smoke.layers.utils import (
    nms_hm,
    select_topk,
    select_point_of_interest,
)


class PostProcessor(nn.Module):
    def __init__(self,
                 smoker_coder,
                 reg_head,        # 8
                 det_threshold,   # 0.25
                 max_detection,   # 50
                 pred_2d,         # True
                 reg_channels,    # (1, 2, 3, 2)
                 convert_onnx,    # False
                 enable_tensorrt, # False
                 cls_num): 
        super(PostProcessor, self).__init__()
        self.smoke_coder = smoker_coder
        self.reg_head = reg_head
        self.det_threshold = det_threshold
        self.max_detection = max_detection
        self.pred_2d = pred_2d
        self.convert_onnx = convert_onnx
        self.enable_tensorrt = enable_tensorrt
        self.reg_channels = reg_channels
        self.cls_num = cls_num

        assert sum(reg_channels) == reg_head, \
            "the sum of {} must be equal to regression channel of {}".format(
                reg_channels, reg_head
            )
        self.dim_channel = get_channel_spec(reg_channels, name="dim")
        self.ori_channel = get_channel_spec(reg_channels, name="ori")

    def prepare_targets(self, targets):
        trans_mat = targets['trans_mat']
        K = targets['K']
        K_src = targets['K_src']
        size = targets['size']
        return dict(trans_mat=trans_mat, K=K, K_src=K_src, size=size)

    def forward(self, predictions, targets):
        #1,1
        #print(predictions[0].shape[0], (targets['trans_mat']).shape[0])
        pred_heatmap, pred_regression = predictions[0], predictions[1]

        if self.enable_tensorrt:
            # (N, C, H, W)
            offset_dims = pred_regression[:, self.dim_channel, ...].clone()
            pred_regression[:, self.dim_channel, ...] = torch.sigmoid(offset_dims) - 0.5

            vector_ori = pred_regression[:, self.ori_channel, ...].clone()
            pred_regression[:, self.ori_channel, ...] = F.normalize(vector_ori)

        if self.convert_onnx:
            return pred_heatmap, pred_regression

        batch = pred_heatmap.shape[0]
        
        # return dict(trans_mat=trans_mat, K=K, K_src=K_src, size=size)
        target_varibales = self.prepare_targets(targets)
        #                                                         50
        # all in [bs, 50]
        scores, indexs, clses, ys, xs = select_topk(pred_heatmap, K=self.max_detection)
        pred_regression = select_point_of_interest(batch, indexs, pred_regression, clses, self.cls_num)
        # bs*50, 8
        pred_regression_pois = pred_regression.view(-1, self.reg_head)
        # bs*50, 2
        pred_proj_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
        
        # FIXME: fix hard code here
        
        # bs*50
        pred_depths_offset = pred_regression_pois[:, 0]
        # bs*50, 2
        pred_proj_offsets = pred_regression_pois[:, sum(self.reg_channels[:1]):sum(self.reg_channels[:2])]
        # bs*50, 3
        pred_dimensions_offsets = pred_regression_pois[:, sum(self.reg_channels[:2]):sum(self.reg_channels[:3])]
        # bs*50, 2
        pred_orientation = pred_regression_pois[:, sum(self.reg_channels[:3]):sum(self.reg_channels)]
        
        # bs*50
        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset, target_varibales["K"])
        pred_locations = self.smoke_coder.decode_location(
            pred_proj_points,
            pred_proj_offsets,
            pred_depths,
            target_varibales["K"],
            target_varibales["trans_mat"]
        )
        pred_dimensions = self.smoke_coder.decode_dimension(clses, pred_dimensions_offsets)
        # we need to change center location to bottom location
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys, pred_alphas = self.smoke_coder.decode_orientation(pred_orientation, pred_locations)

        if self.pred_2d:
            box2d = self.smoke_coder.encode_box2d(
                target_varibales["K_src"],
                pred_rotys,
                pred_dimensions,
                pred_locations,
                target_varibales["size"]
            )
        else:
            box2d = torch.tensor([0, 0, 0, 0])

        # change variables to the same dimension
        clses = clses.view(-1, 1)
        pred_alphas = pred_alphas.view(-1, 1)
        pred_rotys = pred_rotys.view(-1, 1)
        scores = scores.view(-1, 1)
        # change dimension back to h,w,l
        pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)
        result = torch.cat([clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores], dim=1)
        keep_idx = result[:, -1] > self.det_threshold
        #print("***", result.shape)
        result = result[keep_idx]
        #print("***", result.shape)
        return result


def make_smoke_post_processor(cfg):
    smoke_coder = SMOKECoder(
        cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE,
        cfg.MODEL.SMOKE_HEAD.NORMALIZED_FOCAL_REFERENCE,
        cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE,
        cfg.MODEL.DEVICE,
    )

    postprocessor = PostProcessor(
        smoke_coder,
        cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS,
        cfg.TEST.DETECTIONS_THRESHOLD,
        cfg.TEST.DETECTIONS_PER_IMG,
        cfg.TEST.PRED_2D,
        cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL,
        cfg.CONVERT_ONNX,
        cfg.ENABLE_TENSORRT,
        len(cfg.DATASETS.DETECT_CLASSES),
    )

    return postprocessor
