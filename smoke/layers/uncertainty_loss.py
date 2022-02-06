import numpy as np
import torch
from torch.nn import functional as F

def laplacian_aleatoric_uncertainty_loss_original(input, target, log_variance, reduction='mean'):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + 0.5*log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()
"""
def laplacian_aleatoric_uncertainty_loss(input, target, log_std, reduction='sum', reg_weight=None):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss1 = 1.4142 * torch.exp(-log_std) * torch.abs(input - target)
    loss2 = log_std
    loss = loss1 + loss2  
    #if reg_weight is not None:
    #    loss1 *= reg_weight
    #    loss2 *= reg_weight
    #    loss *= reg_weight   
    loss1 = loss1.mean() if reduction == 'mean' else loss1.sum()
    loss2 = loss2.mean() if reduction == 'mean' else loss2.sum()
    loss = loss.mean() if reduction == 'mean' else loss.sum()
    
    return loss, loss1, loss2
"""
def laplacian_aleatoric_uncertainty_loss(input, target, log_std, FUNCTION, reduction='sum', reg_weight=None):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss1 = 1.4142 * torch.exp(-log_std) * FUNCTION( input, target, reduce = False)
    loss2 = log_std
    loss = loss1 + loss2
    #if reg_weight is not None:
    #    loss1 *= reg_weight
    #    loss2 *= reg_weight
    #    loss *= reg_weight
    loss1 = loss1.mean() if reduction == 'mean' else loss1.sum()
    loss2 = loss2.mean() if reduction == 'mean' else loss2.sum()
    loss = loss.mean() if reduction == 'mean' else loss.sum()
    return loss, loss1, loss2

def gaussian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, Neuips'17
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 0.5 * torch.exp(-log_variance) * torch.abs(input - target)**2 + 0.5 * log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()



if __name__ == '__main__':
    pass
