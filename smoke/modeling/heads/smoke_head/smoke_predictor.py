import torch
from torch import nn
from torch.nn import functional as F

from smoke import registry
from smoke.layers.utils import sigmoid_hm
from smoke.modeling.make_layers import _fill_fc_weights

def get_channel_spec(reg_channels, name):
    if name == "dim":
        s = sum(reg_channels[:2])
        e = sum(reg_channels[:3])
    elif name == "ori":
        s = sum(reg_channels[:3])
        e = sum(reg_channels)

    return slice(s, e, 1)


@registry.SMOKE_PREDICTOR.register("SMOKEPredictor")
class SMOKEPredictor(nn.Module):
    # in_channels:64
    def __init__(self, cfg, in_channels):
        super(SMOKEPredictor, self).__init__()
        # 6
        classes = len(cfg.DATASETS.DETECT_CLASSES)
        # 8
        regression = cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
        # (1, 2, 3, 2)  depth, offset, dim, ori
        self.reg_channels = cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL
        # 1
        self.depth_channels = self.reg_channels[0]
        # False
        self.reg_multi_heads = cfg.MODEL.SMOKE_HEAD.REGRESSION_MULTI_HEADS
        # 256
        head_conv = cfg.MODEL.SMOKE_HEAD.NUM_CHANNEL

        assert sum(self.reg_channels) == regression, \
            "the sum of {} must be equal to regression channel of {}".format(
                cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL, cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
            )

        self.dim_channel = get_channel_spec(self.reg_channels, name="dim")
        self.ori_channel = get_channel_spec(self.reg_channels, name="ori")
        # False
        self.convert_onnx = cfg.CONVERT_ONNX

        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            nn.BatchNorm2d(head_conv),

            nn.ReLU(inplace=True),

            nn.Conv2d(head_conv,
                      classes,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )

        # todo: what is datafill here
        self.class_head[-1].bias.data.fill_(-2.19)

        # Specific channel for (depth_offset, keypoint_offset, dimension_offset, orientation)
        if not self.reg_multi_heads:
            self.regression_head = nn.Sequential(
                nn.Conv2d(in_channels,
                          head_conv,
                          kernel_size=3,
                          padding=1,
                          bias=True),
                nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv,
                          regression * classes,
                          kernel_size=1,
                          padding=1 // 2,
                          bias=True)
            )
            # what??
            _fill_fc_weights(self.regression_head)

        else:
            self.depth_head = nn.Sequential(
                nn.Conv2d(in_channels,
                          head_conv,
                          kernel_size=3,
                          padding=1,
                          bias=True),
                nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv,
                          self.reg_channels[0],
                          kernel_size=1,
                          padding=1 // 2,
                          bias=True)
            )
            _fill_fc_weights(self.depth_head)

            self.offset_head = nn.Sequential(
                nn.Conv2d(in_channels,
                          head_conv,
                          kernel_size=3,
                          padding=1,
                          bias=True),
                nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv,
                          self.reg_channels[1],
                          kernel_size=1,
                          padding=1 // 2,
                          bias=True)
            )
            _fill_fc_weights(self.offset_head)

            self.dim_head = nn.Sequential(
                nn.Conv2d(in_channels,
                          head_conv,
                          kernel_size=3,
                          padding=1,
                          bias=True),
                nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv,
                          self.reg_channels[2],
                          kernel_size=1,
                          padding=1 // 2,
                          bias=True)
            )
            _fill_fc_weights(self.dim_head)

            self.orientation_head = nn.Sequential(
                nn.Conv2d(in_channels,
                          head_conv,
                          kernel_size=3,
                          padding=1,
                          bias=True),
                nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv,
                          self.reg_channels[3],
                          kernel_size=1,
                          padding=1 // 2,
                          bias=True)
            )
            _fill_fc_weights(self.orientation_head)


    def forward(self, features):
        head_class = self.class_head(features)

        if not self.reg_multi_heads:
            head_regression = self.regression_head(features)
        else:
            depth_reg = self.depth_head(features)
            offset_reg = self.offset_head(features)
            dim_reg = self.dim_head(features)
            orientation_reg = self.orientation_head(features)
            head_regression = torch.cat([depth_reg, offset_reg, dim_reg, orientation_reg], 1)
        
        # causing diff between train and test? 
        head_class = sigmoid_hm(head_class, self.training)
        
        if not self.convert_onnx:
            # (N, C, H, W)
            offset_dims = head_regression[:, self.dim_channel, ...].clone()
            head_regression[:, self.dim_channel, ...] = torch.sigmoid(offset_dims) - 0.5

            vector_ori = head_regression[:, self.ori_channel, ...].clone()
            head_regression[:, self.ori_channel, ...] = F.normalize(vector_ori)

        return [head_class, head_regression]

# in_channels = 64
def make_smoke_predictor(cfg, in_channels):
    func = registry.SMOKE_PREDICTOR[
        # "SMOKEPredictor"
        cfg.MODEL.SMOKE_HEAD.PREDICTOR
    ]
    return func(cfg, in_channels)
