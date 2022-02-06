import torch
from torch.nn import functional as F
import numpy as np

from smoke.modeling.smoke_coder import SMOKECoder
from smoke.layers.focal_loss import FocalLoss
from smoke.layers.gaussian_focal_loss import GaussianFocalLoss
from smoke.layers.generalized_focal_loss import GeneralizedFocalLoss
from smoke.layers.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
from smoke.layers.utils import select_point_of_interest
# from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import BboxOverlaps3D

"""
        smoke_coder,
        cls_loss=  focal_loss,     input: prediction, target, cls_weights
        reg_loss=  "DisL1",
        loss_weight=  (1., 1.),
        max_objs=  100,
        reg_channels = (1, 2, 3, 2),
"""
"""
waymo"
        smoke_coder,
        cls_loss=  focal_loss,     input: prediction, target, cls_weights
        reg_loss=  "DisL1",
        loss_weight=  (1., 2.),
        max_objs=  100,
        reg_channels = (2, 2, 3, 2),
"""
class SMOKELossComputation():
    def __init__(self,
                 smoke_coder,
                 cls_loss,
                 reg_loss,
                 loss_weight,
                 max_objs,
                 reg_channels,
                 cfg):
        self.smoke_coder = smoke_coder
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.loss_weight = loss_weight
        self.max_objs = max_objs
        self.reg_channels = reg_channels
        self.depth_channels = reg_channels[0]
        self.vertex_num = 8.0
        self.cls_num = len(cfg.DATASETS.DETECT_CLASSES)
        # self.iou_calculator = BboxOverlaps3D(coordinate="camera")
    # pred_iou_box:       N,3,8
    # target_iou_box:     N,3,8
    # reg_mask:           N,3,8
    # divide & weight
    def bev_iou_loss(self, pred_iou_box, target_iou_box, reg_mask, divide, weight):
        # bs*100
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_iou_box[:,0,0], pred_iou_box[:,2,0],pred_iou_box[:,0,4], pred_iou_box[:,2,4]
        b2_x1, b2_y1, b2_x2, b2_y2 = target_iou_box[:,0,0], target_iou_box[:,2,0],target_iou_box[:,0,4], target_iou_box[:,2,4]
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()) * torch.max(
                inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        # bs*100
        iou_loss = (1 - inter_area / (b1_area + b2_area - inter_area))*reg_mask
        iou_loss = torch.sum(iou_loss)/divide*weight
        return iou_loss

    def l1_loss(self, inputs, targets, weight, reduction="sum"):
        diffs = torch.abs(inputs - targets) * weight
        if reduction == "sum":
            loss = torch.sum(diffs)
        elif reduction == "mean":
            loss = torch.mean(diffs)
        return loss
    def prepare_targets(self, targets):
        cls_weights = targets['cls_weights']
        heatmaps = targets['heatmaps']
        regression = targets['regression']
        cls_ids = targets['cls_ids']
        proj_points = targets['proj_points']
        dimensions = targets['dimensions']
        locations = targets['locations']
        rotys = targets['rotys']
        trans_mat = targets['trans_mat']
        K = targets['K']
        reg_mask = targets['reg_mask']
        iou_box = targets['iou_box']
        flip_mask = targets['flip_mask']
        conner_2d = targets['conner_2d']
        p_offsets = targets['p_offsets']
        return heatmaps, regression, dict(cls_ids=cls_ids,
                                          proj_points=proj_points,
                                          dimensions=dimensions,
                                          locations=locations,
                                          rotys=rotys,
                                          trans_mat=trans_mat,
                                          K=K,
                                          reg_mask=reg_mask,
                                          cls_weights=cls_weights,
                                          iou_box = iou_box,
                                          flip_mask=flip_mask,
                                          conner_2d=conner_2d,
                                          p_offsets=p_offsets)
    """
    def prepare_targets(self, targets):
        # bs,6
        cls_weights = torch.stack([t.get_field("cls_weights") for t in targets])
        # bs,6,h,w
        heatmaps = torch.stack([t.get_field("hm") for t in targets])
        # bs, 100, 3, 8    3d box
        regression = torch.stack([t.get_field("reg") for t in targets])
        # bs, 100
        cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
        # bs, 100, 2  2d center in (int) in 120*160 image plane
        proj_points = torch.stack([t.get_field("proj_p") for t in targets])
        # bs, 100, 3
        dimensions = torch.stack([t.get_field("dimensions") for t in targets])
        # bs, 100, 3
        locations = torch.stack([t.get_field("locations") for t in targets])
        # bs, 100
        rotys = torch.stack([t.get_field("rotys") for t in targets])
        # bs, 3, 3 down sample matrix from 480*640 to 120*160
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets])
        # bs, 3, 3 changed K because of augmentation
        K = torch.stack([t.get_field("K") for t in targets])
        # bs, 100 0/1
        reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
        # bs, 100, 3, 8
        iou_box = torch.stack([t.get_field("iou_box") for t in targets])

        flip_mask = torch.stack([t.get_field("flip_mask") for t in targets])

        conner_2d = torch.stack([t.get_field("conner_2d") for t in targets])

        return heatmaps, regression, dict(cls_ids=cls_ids,
                                          proj_points=proj_points,
                                          dimensions=dimensions,
                                          locations=locations,
                                          rotys=rotys,
                                          trans_mat=trans_mat,
                                          K=K,
                                          reg_mask=reg_mask,
                                          cls_weights=cls_weights,
                                          iou_box = iou_box,
                                          flip_mask=flip_mask,
                                          conner_2d=conner_2d)
    """
    # targets_variables:
    #          cls_ids: bs, 100
    #          proj_points: bs, 100, 2  2d center in (int) in 120*160 image plane
    #          dimensions: bs, 100, 3
    #          locations: bs, 100, 3
    #          rotys: bs, 100
    #          trans_mat: bs, 3, 3 down sample matrix from 480*640 to 120*160
    #          K: bs, 3, 3 changed K because of augmentation
    #          reg_mask: bs, 100 0/1
    #          cls_weights: bs,6
    #pred_regression: bs, 9 * 6 , h, w
    def prepare_predictions(self, targets_variables, pred_regression):
        batch, channel = pred_regression.shape[0], pred_regression.shape[1]
        channel = int(channel/self.cls_num)
        # bs, 100, 2  2d center in (int) in 120*160 image plane
        targets_proj_points = targets_variables["proj_points"]
        target_p_offset = targets_variables["p_offsets"]
        target_p_offset = target_p_offset.view(-1,2)
        target_depth = targets_variables["locations"][:,:,2] 
        target_depth = target_depth.view(-1) # bs*100
        #target_depth = self.smoke_coder.decode_target_depth(target_depth, targets_variables["K"])
        # obtain prediction from points of interests
        # bs, 100, 8
        target_cls = targets_variables["cls_ids"]
        pred_regression_pois = select_point_of_interest(batch, targets_proj_points, pred_regression, target_cls, self.cls_num)
        # bs * 100, 8
        pred_regression_pois = pred_regression_pois.view(-1, channel)

        # FIXME: fix hard code here
        # Specific channel for (depth_offset, keypoint_offset, dimension_offset, orientation)
        if self.depth_channels == 1:
            # bs * 100
            pred_depths_offset = pred_regression_pois[:, 0]
            # bs * 100, 2
            pred_proj_offsets = pred_regression_pois[:, sum(self.reg_channels[:1]):sum(self.reg_channels[:2])]
            # bs * 100, 3
            pred_dimensions_offsets = pred_regression_pois[:, sum(self.reg_channels[:2]):sum(self.reg_channels[:3])]
            # bs * 100, 2
            pred_orientation = pred_regression_pois[:, sum(self.reg_channels[:3]):sum(self.reg_channels)]
            # bs * 100    in pixel
            pred_depths = self.smoke_coder.decode_depth(pred_depths_offset, targets_variables["K"])
            pred_depths_std = None
        else:
            pred_depths_offset, pred_depths_log_std = pred_regression_pois[:, 0], pred_regression_pois[:, 1]
            pred_proj_offsets = pred_regression_pois[:, sum(self.reg_channels[:1]):sum(self.reg_channels[:2])]
            pred_dimensions_offsets = pred_regression_pois[:, sum(self.reg_channels[:2]):sum(self.reg_channels[:3])]
            pred_orientation = pred_regression_pois[:, sum(self.reg_channels[:3]):sum(self.reg_channels)]

            pred_depths = self.smoke_coder.decode_depth(pred_depths_offset, targets_variables["K"])
            # for waymo 
            pred_depths_std = torch.exp(pred_depths_log_std)
            # this means the log of depth std
            pred_depths_std = self.smoke_coder.decode_depth_std(pred_depths_std, targets_variables["K"])
            
        #  bs * 100, 3 (xyz in the real world) 
        #  the center!!!!  not the top of car!!!!!
        pred_locations_from_xy = self.smoke_coder.decode_location(
            targets_proj_points,
            pred_proj_offsets,
            target_depth,
            targets_variables["K"],
            targets_variables["trans_mat"]
        )
        pred_locations_from_depth = self.smoke_coder.decode_location(
            targets_proj_points,
            target_p_offset,
            pred_depths,
            targets_variables["K"],
            targets_variables["trans_mat"]
        )
        #  bs, 3  (lx ly lz in the real world)
        #print(targets_variables["cls_ids"].device, pred_dimensions_offsets.device)
        pred_dimensions = self.smoke_coder.decode_dimension(
            targets_variables["cls_ids"],
            pred_dimensions_offsets,
        )
        # we need to change center location to bottom location
        # changed to the top car!!!
        # because the camera is on the top of the car????
        pred_locations_from_xy[:, 1] += pred_dimensions[:, 1] / 2
        pred_locations_from_depth[:, 1] += pred_dimensions[:, 1] / 2
        
        
        # rotys:  batzh_size * max_points, theta in the paper
        # alphas: batzh_size * max_points, alphaz in the paper
        pred_rotys, pred_alpha = self.smoke_coder.decode_orientation(
            pred_orientation,
            targets_variables["locations"],
            targets_variables["flip_mask"]
        )
        
        #bs * 100, 3, 8   --->   x = ori
        pred_box3d_rotys = self.smoke_coder.encode_box3d(
            pred_rotys,
            targets_variables["dimensions"],
            targets_variables["locations"]
        )
        
        #bs * 100, 3, 8  --->   x = dim_offset
        pred_box3d_dims = self.smoke_coder.encode_box3d(
            targets_variables["rotys"],
            pred_dimensions,
            targets_variables["locations"]
        )
        
        #bs * 100, 3, 8  --->   x = depth,x,y_offset
        pred_box3d_locs_xy = self.smoke_coder.encode_box3d(
            targets_variables["rotys"],
            targets_variables["dimensions"],
            pred_locations_from_xy
        )
        
        pred_box3d_locs_depth = self.smoke_coder.encode_box3d(
            targets_variables["rotys"],
            targets_variables["dimensions"],
            pred_locations_from_depth
        )
        """
        #bs * 100, 3, 8
        pred_box3d_iou = self.smoke_coder.encode_box3d(
            torch.zeros_like(targets_variables["rotys"]),
            pred_dimensions,
            pred_locations
        )
        """
        if self.reg_loss == "DisL1":
            return dict(ori=pred_box3d_rotys, # x = ori                bs * 100, 3, 8
                        dim=pred_box3d_dims,  # x = dim_offset         bs * 100, 3, 8
                        loc_xy=pred_box3d_locs_xy,  # x = depth,x,y_offset   bs * 100, 3, 8
                        loc_depth=pred_box3d_locs_depth,
                        #iou_box = pred_box3d_iou,
                        depth_std = pred_depths_std,) #None for jdx, pred_depths_std for waymo
        return None

    def encode_conner_2d(self, box3d, K):
        # N,3,8
        corners_2d = torch.matmul(K, box3d)
        corners_2d = corners_2d[:, :2, :] / (corners_2d[:, 2, :].view(
            box3d.shape[0], 1, box3d.shape[2]) + 1e-7)
        return corners_2d

    def prepare_loss_weight(self, targets_variables, predict_boxes3d, pred_heatmap, reg_mask):

        # cls score
        targets_proj_points = targets_variables["proj_points"]
        batch = pred_heatmap.shape[0]
        cls_scores = select_point_of_interest(batch, targets_proj_points, pred_heatmap)
        cls_scores = torch.max(cls_scores, -1)[0].view(-1, 1)

        # box iou
        gt_locations = targets_variables["locations"].view(-1, 3)
        gt_rotys =targets_variables["rotys"].view(-1, 1)
        gt_dimensions = targets_variables["dimensions"].view(-1, 3)
        gt_dimensions = gt_dimensions[:, [1, 2, 0]]  # lhw -> hwl
        gt_boxes = torch.cat((gt_locations, gt_dimensions, gt_rotys), dim=-1)

        pred_locations = predict_boxes3d["pred_locs"]
        pred_rotys = predict_boxes3d["pred_rotys"].unsqueeze(-1)
        pred_dimensions = predict_boxes3d["pred_dims"]
        pred_dimensions = pred_dimensions[:, [1, 2, 0]] # lhw -> hwl
        pred_boxes = torch.cat((pred_locations, pred_dimensions, pred_rotys), dim=-1)

        bbox_ious_3d = self.iou_calculator(gt_boxes, pred_boxes, "iou")
        nums_boxes = bbox_ious_3d.shape[0]
        bbox_ious_3d = bbox_ious_3d[range(nums_boxes), range(nums_boxes)].view(-1, 1)

        num_objs = torch.sum(reg_mask)
        reg_weight = 2 * cls_scores + (1 - bbox_ious_3d)
        reg_weight_masked = reg_weight[reg_mask.bool()]
        reg_weight_masked = F.softmax(reg_weight_masked, dim=0) * num_objs
        reg_weight[reg_mask.bool()] = reg_weight_masked
        reg_weight = reg_weight.view(-1, 1, 1)
        return reg_weight

    def __call__(self, predictions, targets):
        pred_heatmap, pred_regression = predictions[0], predictions[1]
        """
        targets_heatmap: bs,6,h,w                                                     gaussed heatmap
        targets_regression: bs, 100, 3, 8/9    3d box                                   gennerated from the annotation
        targets_variables:
              cls_ids: bs, 100
              proj_points: bs, 100, 2  2d center in (int) in 120*160 image plane
              dimensions: bs, 100, 3
              locations: bs, 100, 3
              rotys: bs, 100
              trans_mat: bs, 3, 3 down sample matrix from 480*640 to 120*160
              K: bs, 3, 3 changed K because of augmentation
              reg_mask: bs, 100 0/1
              cls_weights: bs,6
              iou_box: bs, 100, 3, 8
        """
        # same for jdx and waymo
        targets_heatmap, targets_regression, targets_variables = self.prepare_targets(targets)

        # dict(ori=pred_box3d_rotys, # x = ori                bs * 100, 3, 8
        #      dim=pred_box3d_dims,  # x = dim_offset         bs * 100, 3, 8
        #      loc=pred_box3d_locs,  # x = depth,x,y_offset   bs * 100, 3, 8
        #      depth_std = pred_depths_std,) #None for jdx, pred_depths_std(bs * 100) for waymo
        predict_boxes3d = self.prepare_predictions(targets_variables, pred_regression)
        

        hm_loss = self.cls_loss(pred_heatmap, targets_heatmap, targets_variables["cls_weights"]) * self.loss_weight[0]
        
        # bs * 100, 3, 8
        targets_regression = targets_regression.view(-1, targets_regression.shape[2], targets_regression.shape[3])
        
        ###########  yzk add for weighted regress  ###########  
        #reg_weight = targets_variables["locations"].view(-1, 3)  # bs*100, 3
        #reg_weight = reg_weight[:,2] # bs*100
        #reg_weight = 0.06667 * reg_weight  # bs*100
        #reg_weight = reg_weight.view(-1, 1, 1)
        #reg_weight = reg_weight.expand_as(targets_regression)
        #print(targets_variables["locations"][0,0,2].data, reg_weight[0,0,0].data)
        ###########  yzk add for weighted regress  ###########

        """
        use_iou_loss = False
        use_conner_loss = False
        use_smooth_l1 = False
        """
        
        # bs * 100
        reg_mask = targets_variables["reg_mask"].flatten()
        #if self.reg_loss == "WeightedDisL1":
        #    reg_weight = self.prepare_loss_weight(targets_variables, predict_boxes3d, pred_heatmap, reg_mask)
        #    reg_weight = reg_weight.expand_as(targets_regression)
        valid_obj_num = reg_mask.sum()
        # bs * 100, 1, 1
        reg_mask = reg_mask.view(-1, 1, 1)
        # bs * 100, 3, 8
        reg_mask = reg_mask.expand_as(targets_regression)

        """
        # bs * 100
        cls_ids = targets_variables["cls_ids"].flatten()
        TMP = torch.ones_like(cls_ids, dtype=torch.float32)
        TMP = TMP.to(cls_ids.device)
        tmp = torch.ones_like(cls_ids, dtype=torch.float32)/5
        tmp = tmp.to(cls_ids.device)
        weight = torch.where(cls_ids==0, TMP, tmp)  # bs * 100
        weight = weight.view(-1, 1, 1)
        weight = weight.expand_as(targets_regression)
        #print(weight[0,0,0].data)
        """

        """
        iou_mask = targets_variables["reg_mask"].flatten()
        target_iou_box = targets_variables["iou_box"]
        # bs * 100, 3, 8
        target_iou_box = target_iou_box.view(-1, target_iou_box.shape[2], target_iou_box.shape[3])
        bev_iou_loss = self.bev_iou_loss(predict_boxes3d["iou_box"], target_iou_box, iou_mask, valid_obj_num, self.loss_weight[1]) * 5
        """
        """
        target_conner = targets_variables["conner_2d"] 
        target_conner = target_conner.view(-1, target_conner.shape[2], target_conner.shape[3])
        # bs, 3, 3
        K = targets_variables["K"] 
        N_batch = K.shape[0]
        N = target_conner.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()
        K = K[obj_id]   # bs * 100, 3, 3
        conner_mask = targets_variables["reg_mask"].flatten()
        conner_mask = conner_mask.view(-1, 1, 1)
        conner_mask = conner_mask.expand_as(target_conner)
        conner_ori = self.encode_conner_2d(predict_boxes3d["ori"], K)
        conner_dim = self.encode_conner_2d(predict_boxes3d["dim"], K)
        conner_loc = self.encode_conner_2d(predict_boxes3d["loc"], K)
        #print(target_conner[0])
        #print(conner_ori[0])
        #print("\n")
        #conner_ori = predict_boxes3d["ori"][:,0:2,:]
        #conner_dim = predict_boxes3d["ori"][:,0:2,:]
        #conner_loc = predict_boxes3d["ori"][:,0:2,:]
        if use_smooth_l1:
            FUNCTION = F.smooth_l1_loss
        else:
            FUNCTION = F.l1_loss
            
        loss_conner_ori = FUNCTION( conner_ori * conner_mask, target_conner * conner_mask,
                    reduction = "sum") / (self.vertex_num * valid_obj_num * 2) * self.loss_weight[1]
        loss_conner_dim = FUNCTION( conner_dim * conner_mask, target_conner * conner_mask,
                    reduction = "sum") / (self.vertex_num * valid_obj_num * 2) * self.loss_weight[1]
        loss_conner_loc = FUNCTION( conner_loc * conner_mask, target_conner * conner_mask,
                    reduction = "sum") / (self.vertex_num * valid_obj_num * 2) * self.loss_weight[1]
        """
        FUNCTION = F.l1_loss
        
        reg_loss_ori = FUNCTION(
            predict_boxes3d["ori"] * reg_mask,
            targets_regression * reg_mask,
            reduction = "sum") / (self.vertex_num * valid_obj_num) * self.loss_weight[1]
        reg_loss_dim = FUNCTION(
            predict_boxes3d["dim"] * reg_mask,
            targets_regression * reg_mask,
            reduction = "sum") / (self.vertex_num * valid_obj_num) * self.loss_weight[1]
        reg_loss_xy = FUNCTION(
            predict_boxes3d["loc_xy"] * reg_mask,
            targets_regression * reg_mask,
            reduction = "sum") / (self.vertex_num * valid_obj_num) * self.loss_weight[1]
        reg_loss_depth = FUNCTION(
            predict_boxes3d["loc_depth"] * reg_mask,
            targets_regression * reg_mask,
            reduction = "sum") / (self.vertex_num * valid_obj_num) * self.loss_weight[1]
        """
        if self.depth_channels == 1:
            reg_loss_loc = FUNCTION(
                predict_boxes3d["loc"] * reg_mask,
                targets_regression * reg_mask,
                reduction = "sum") / (self.vertex_num * valid_obj_num * 3) * self.loss_weight[1]
        else:
            reg_loss_xy = FUNCTION(
                # bs * 100, 3, 8
                predict_boxes3d["loc"][:, 0:2, :] * reg_mask[:, 0:2, :],
                targets_regression[:, 0:2, :] * reg_mask[:, 0:2, :],
                reduction = "sum") / (self.vertex_num * valid_obj_num * 2) * self.loss_weight[1]
            # bs * 100, 1
            depth_std = predict_boxes3d["depth_std"].view(-1, 1)
            # bs * 100, 8
            depth_std = depth_std.expand_as(reg_mask[:, 2, :])
            depth_log_std = torch.log(depth_std)
            reg_loss_z, loss1 ,loss2 = laplacian_aleatoric_uncertainty_loss(
                # bs * 100,  8    component z of pred 3D boxes
                predict_boxes3d["loc"][:, 2, :] * reg_mask[:, 2, :],
                # bs * 100,  8    component z of gt 3D boxes
                targets_regression[:, 2, :] * reg_mask[:, 2, :],
                # bs * 100,  8
                depth_log_std * reg_mask[:, 2, :] ,
                FUNCTION,
                reduction="sum",)

            loss1 = loss1 / (self.vertex_num * valid_obj_num) * self.loss_weight[1]
            loss2 = loss2 / (self.vertex_num * valid_obj_num) * self.loss_weight[1]
            reg_loss_z = reg_loss_z / (self.vertex_num * valid_obj_num) * self.loss_weight[1]

            reg_loss_loc = reg_loss_xy + reg_loss_z
        """

        reg_loss_details = dict(reg_ori=reg_loss_ori, reg_dim=reg_loss_dim, reg_xy=reg_loss_xy, reg_depth=reg_loss_depth)
        LOSS = reg_loss_ori + reg_loss_dim + reg_loss_xy + reg_loss_depth
        """
        if use_iou_loss:
            #reg_loss_details["bev_iou"] = 1 - bev_iou_loss/2
            reg_loss_details["bev_iou"] = bev_iou_loss
            LOSS += bev_iou_loss
        if use_conner_loss:
            reg_loss_details["conner_ori"] = loss_conner_ori
            reg_loss_details["conner_dim"] = loss_conner_dim
            reg_loss_details["conner_loc"] = loss_conner_loc
            LOSS += (loss_conner_ori + loss_conner_dim + loss_conner_loc)
        """
        return hm_loss, LOSS, reg_loss_details


def make_smoke_loss_evaluator(cfg):
    smoke_coder = SMOKECoder(
        # (28.01, 16.32)
        cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE,
        # 720.0
        cfg.MODEL.SMOKE_HEAD.NORMALIZED_FOCAL_REFERENCE,

        # ((4.392, 1.658, 1.910),
        # (1.773, 1.525, 0.740),
        # (0.505, 1.644, 0.582),
        # (7.085, 2.652, 2.523),
        # (2.790, 1.651, 1.201),
        # (8.208, 2.869, 2.645))
        cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE,
        # "cuda"
        cfg.MODEL.DEVICE,
    )

    if cfg.MODEL.SMOKE_HEAD.LOSS_TYPE[0] == "FocalLoss":
        focal_loss = FocalLoss(
            cfg.MODEL.SMOKE_HEAD.LOSS_ALPHA,
            cfg.MODEL.SMOKE_HEAD.LOSS_BETA,
        )
    if cfg.MODEL.SMOKE_HEAD.LOSS_TYPE[0] == "GaussianFocalLoss":
        focal_loss = GaussianFocalLoss(
            cfg.MODEL.SMOKE_HEAD.LOSS_ALPHA,
            cfg.MODEL.SMOKE_HEAD.LOSS_BETA,
        )
    if cfg.MODEL.SMOKE_HEAD.LOSS_TYPE[0] == "GeneralizedFocalLoss":
        focal_loss = GeneralizedFocalLoss(
            cfg.MODEL.SMOKE_HEAD.GENERALIZED_FL_BELTA,
        )

    loss_evaluator = SMOKELossComputation(
        smoke_coder,
        cls_loss=focal_loss,
        reg_loss=cfg.MODEL.SMOKE_HEAD.LOSS_TYPE[1],
        loss_weight=cfg.MODEL.SMOKE_HEAD.LOSS_WEIGHT,
        max_objs=cfg.DATASETS.MAX_OBJECTS,
        reg_channels = cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL,
        cfg=cfg,
    )

    return loss_evaluator
