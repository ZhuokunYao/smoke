import torch
from torch import nn

from smoke.structures.image_list import to_image_list
from ..preprocess import build_preprocess
from ..backbone import build_backbone
from ..heads.heads import build_heads


class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()
        #input range:[0,255]    /255, -mean, /std 
        self.preprocess = build_preprocess(cfg)
        #down ration:4     output_channels:64
        self.backbone = build_backbone(cfg)
        #train:   return {}, dict(cls_loss=loss_heatmap, reg_loss=loss_regression), reg_loss_details
           # input: prediction, targets_train
           # output: hm_loss, reg_loss_ori + reg_loss_dim + reg_loss_loc, reg_loss_details(dict of ori & dim & loc)
        #test:    return prediction, {}, {}
           # input: prediction, targets_test
           #                                                   shifted!!       top car!!            filtered by this
           # result = torch.cat([clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores], dim=1)
        self.heads = build_heads(cfg, self.backbone.out_channels)

    #def forward(self, images, K_src=None, size=None, cls_weights=None, heatmaps=None, regression=None,
    #                          cls_ids=None, proj_points=None, dimensions=None, locations=None, 
    #                          rotys=None, trans_mat=None, K=None, reg_mask=None, iou_box=None, 
    #                          flip_mask=None, conner_2d=None):
    def forward(self, images, target):
        """
        Args:
            images:
            targets:
        Returns:

        """
        #print(target['rotys'].device, target['heatmaps'].device)
        #images = to_image_list(images)
        #print(targets[0].get_field("cls_weights").device)
        
        #images_pre = self.preprocess(images.tensors)
        #print(cls_weights.device)
        images_pre = self.preprocess(images)
        features = self.backbone(images_pre)
        result, detector_losses, detector_reg_loss_details = self.heads(features, target)

        if self.training:
            losses, reg_loss_details = {}, {}
            losses.update(detector_losses)
            reg_loss_details.update(detector_reg_loss_details)

            return losses, reg_loss_details

        return result