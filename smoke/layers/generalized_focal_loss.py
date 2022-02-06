import torch
from torch import nn

"""
loss = - (gt-y)^2 * gtcls_weight * [gt*logy + (1-gt)log(1-y) ]
"""
class GeneralizedFocalLoss(nn.Module):

    def __init__(self, beta=2):
        super(GeneralizedFocalLoss, self).__init__()
        self.beta = beta

    def forward(self, prediction, target, cls_weights):
        # N,C  --->   N,C,H,W
        cls_weights = cls_weights.unsqueeze(-1).unsqueeze(-1)
        shape= prediction.shape
        cls_weights = cls_weights.repeat(1, 1, shape[2], shape[3])

        loss = 0.
        positive_index = target.eq(1).float()
        num_positive = positive_index.float().sum()
        scale_factor = torch.pow((target - prediction).abs(), self.beta)
        loss -= scale_factor  * (target*torch.log(prediction) + (1-target)*torch.log(1-prediction)) * cls_weights
        num_positive = max(1.0, num_positive)
        loss = loss.sum() / num_positive

        return loss