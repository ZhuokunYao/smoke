import torch
from torch import nn

#
class Normalization(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super(Normalization, self).__init__()
        self.mean = mean
        self.std = std
        # use 1*1 kernel to achieve image normalization 
        self.conv1x1 = nn.Conv2d(in_channels=3,
                                 out_channels=3,
                                 kernel_size=1,
                                 bias=True)
        # [channel_out, channel_in, ksize, ksize]
        self.kernel = torch.zeros(size=(3, 3, 1, 1), dtype=torch.float32,
                                  requires_grad=False)
        self.bias = torch.zeros(size=(3,), dtype=torch.float32,
                                requires_grad=False)
        for i in range(3):
            self.kernel[i, i, 0, 0] = 1.0 / (self.std[i] * 255.0)
            self.bias[i] = -self.mean[i] / self.std[i]

        self.conv1x1.weight.data.copy_(self.kernel)
        self.conv1x1.bias.data.copy_(self.bias)
        for param in self.conv1x1.parameters():
            param.requires_grad = False

    def forward(self, x):
        #print("x: ", x.device)
        #print("weight: ", self.conv1x1.weight.device)
        x = self.conv1x1(x)
        # print(self.conv1x1.weight.data)
        # print(self.conv1x1.bias.data)
        return x

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
