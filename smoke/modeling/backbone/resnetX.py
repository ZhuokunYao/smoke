from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps = 0.001)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class BasicBlock(nn.Module):
    expansion = 1
    depth = 2
    def __init__(self, inplanes, cfg, stride=1, downsample=None):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg[0], stride)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg[0], cfg[1])
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    depth = 3
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetX(nn.Module):

    def __init__(self, block, layers, cfg = None):
        self.inplanes = 64
        self.deconv_with_bias = False
        super(ResNetX, self).__init__()
        if cfg is None or len(cfg) == 0:
            cfg = [[3, 64], [64] * layers[0] * block.depth,
                   [128] * (layers[1] * block.depth + 1),
                   [256] * (layers[1] * block.depth + 1),
                   [512] * (layers[1] * block.depth + 1),
                   [256, 128, 64]]
            cfg = [item for sub_list in cfg for item in sub_list]
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(cfg[1], momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        count = 2
        self.layer1 = self._make_layer(block, cfg[count-1], layers[0], cfg[count:count+layers[0] * block.depth])
        count += layers[0] * block.depth
        self.layer2 = self._make_layer(block, cfg[count-1], layers[1], cfg[count:count + layers[1] * block.depth +1], stride = 2)
        count += layers[1] * block.depth + 1
        self.layer3 = self._make_layer(block, cfg[count-1], layers[2], cfg[count:count + layers[2] * block.depth +1], stride = 2)
        count += layers[2] * block.depth + 1
        self.layer4 = self._make_layer(block, cfg[count-1], layers[3], cfg[count:count + layers[3] * block.depth +1], stride = 2)
        count += layers[3] * block.depth + 1
        self.deconv_layers = self._make_deconv_layer(cfg[count-1], 3, cfg[count:], [4, 4, 4])


    def _make_layer(self, block, inplanes, blocks, cfg, stride = 1):
        downsample = None
        if blocks * block.depth == len(cfg):
            outplanes = cfg[block.depth - 1]
        else:
            outplanes = cfg[block.depth]
        if stride != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        layers = []
        layers.append(block(inplanes, cfg[:block.depth], stride, downsample))

        inplanes = cfg[block.depth-1]
        if blocks * block.depth == len(cfg):
            for i in range(1, blocks):
                layers.append(block(inplanes, cfg[block.depth * i:block.depth * (i + 1)]))
        else:
            for i in range(1, blocks):
                layers.append(block(inplanes, cfg[block.depth * i +1 :block.depth * (i + 1)+1]))

        return nn.Sequential(*layers)


    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, inplanes, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)

        return x
    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            #pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')

# @registry.BACKBONES.register('resnetx34')
# def resnetx34(cfg):
#     """Constructs a ResNet-34 model.
#     """
#     model = ResNetX(BasicBlock, [3, 4, 6, 3, 3, 3], cfg = cfg.MODEL.BACKBONE.CONFIG)
#
#     if not cfg.SOLVER.PRETRAIN_MODEL:
#         # load the pretrain backbone model which is trained in classification, not detection
#         if len(cfg.MODEL.BACKBONE.PRETRAIN_MODEL) > 0:
#             model.init_from_pretrain(torch.load(cfg.MODEL.BACKBONE.PRETRAIN_MODEL))
#         else:
#             model.init_from_pretrain(load_state_dict_from_url(model_urls['resnet34']))
#
#     return model
#
# @registry.BACKBONES.register('resnetx18')
# def resnetx18(cfg):
#     """Constructs a ResNet-18 model.
#     """
#     model = ResNetX(BasicBlock, [2, 2, 2, 2, 2, 2], cfg = cfg.MODEL.BACKBONE.CONFIG)
#
#     if not cfg.SOLVER.PRETRAIN_MODEL:
#         # load the pretrain backbone model which is trained in classification, not detection
#         if len(cfg.MODEL.BACKBONE.PRETRAIN_MODEL) > 0:
#             model.init_from_pretrain(torch.load(cfg.MODEL.BACKBONE.PRETRAIN_MODEL))
#         else:
#             model.init_from_pretrain(load_state_dict_from_url(model_urls['resnet18']))
#
#     return model

_STAGE_SPECS = {"RESNETX-18": (BasicBlock, [2, 2, 2, 2], 18),
                "RESNET-34": (BasicBlock, [3, 4, 6, 3], 34),
                "RESNET-50": (Bottleneck, [3, 4, 6, 3], 50),
                "RESNET-101": (Bottleneck, [3, 4, 23, 3], 101),
                "RESNET-152": (Bottleneck, [3, 8, 36, 3], 152)}

def get_resnet(cfg):
  block_class, layers, num_layers = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]

  model = ResNetX(block_class, layers, cfg.MODEL.BACKBONE.CHANNELS)

  # if not os.path.exists(cfg.SOLVER.PRETRAIN_MODEL):
  #   model.init_weights(num_layers, pretrained=True)
  return model

if __name__ == '__main__':
  cfg = [3, 64, 34, 64, 49, 64, 106, 86, 86, 94, 86, 166, 163, 163, 82, 163, 90, 423, 423, 115, 423, 256, 128, 64]
  # model = ResNet(BasicBlock, [3, 4, 6, 3, 3, 3], cfg = cfg)
  model = ResNetX(BasicBlock, [2, 2, 2, 2], cfg = cfg)
  input = torch.autograd.Variable(torch.randn(1, 3, 384, 1280))
  output = model(input)
  for layer in output:
    print(layer.shape)