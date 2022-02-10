import torch
from torch import nn
from torch.nn import functional as F

from smoke import registry
from smoke.layers.utils import sigmoid_hm
from smoke.modeling.make_layers import _fill_fc_weights

from detectron2.layers import Conv2d, cat, get_norm

from tridet.layers.normalization import ModuleListDial

EPS = 1e-7

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

        ############################### from dd3d ###############################
        norm = 'FrozenBN'
        num_convs = 4
        self.num_levels = 5

        box3d_tower = []
        for i in range(num_convs):
            if norm in ("BN", "FrozenBN"):
                # Each FPN level has its own batchnorm layer.
                # "BN" is converted to "SyncBN" in distributed training (see train.py)
                norm_layer = ModuleListDial([get_norm(norm, in_channels) for _ in range(self.num_levels)])
            else:
                norm_layer = get_norm(norm, in_channels)
            box3d_tower.append(
                Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=norm_layer is None,
                    norm=norm_layer,
                    activation=F.relu
                )
            )
        self.add_module('box3d_tower', nn.Sequential(*box3d_tower))
        # {box3d_tower: 4 conv layers}


        head_configs = {'cls': 4}
        self._version = "v2"

        for head_name, num_convs in head_configs.items():
            tower = []
            if self._version == "v1":
                for _ in range(num_convs):
                    conv_func = nn.Conv2d
                    tower.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
                    if norm == "GN":
                        raise NotImplementedError()
                    elif norm == "NaiveGN":
                        raise NotImplementedError()
                    elif norm == "BN":
                        tower.append(ModuleListDial([nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)]))
                    elif norm == "SyncBN":
                        raise NotImplementedError()
                    tower.append(nn.ReLU())
            elif self._version == "v2":
                for _ in range(num_convs):
                    if norm in ("BN", "FrozenBN"):
                        # Each FPN level has its own batchnorm layer.
                        # "BN" is converted to "SyncBN" in distributed training (see train.py)
                        norm_layer = ModuleListDial([get_norm(norm, in_channels) for _ in range(self.num_levels)])
                    else:
                        norm_layer = get_norm(norm, in_channels)
                    tower.append(
                        Conv2d(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=norm_layer is None,
                            norm=norm_layer,
                            activation=F.relu
                        )
                    )
            else:
                raise ValueError(f"Invalid FCOS2D version: {self._version}")
            self.add_module(f'{head_name}_tower', nn.Sequential(*tower))
        # {cls_tower: 4 conv layers respectively}
        ############################### from dd3d ###############################

        self.class_head = nn.Sequential(
            self.cls_tower,
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
                self.box3d_tower,
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
        features = features['p3']
        head_class = self.class_head(features)
        #print(head_class.shape)
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
