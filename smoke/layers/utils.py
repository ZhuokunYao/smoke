import torch
from torch.nn import functional as F


def sigmoid_hm(hm_features, training=False):
    x = hm_features.sigmoid_()
    if training:
        x = x.clamp(min=1e-4, max=1 - 1e-4)
    return x


def nms_hm(heat_map, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat_map,
                        kernel_size=(kernel, kernel),
                        stride=1,
                        padding=pad)
    eq_index = torch.floor(heat_map - hmax) + 1.0

    return heat_map * eq_index


def select_topk(heat_map, K=100):
    '''
    Args:
        heat_map: heat_map in [N, C, H, W]
        K: top k samples to be selected
        score: detection threshold

    Returns:

    '''
    batch, cls, height, width = heat_map.size()

    # First select topk scores in all classes and batchs
    # [N, C, H, W] -----> [N, C, H*W]
    heat_map = heat_map.view(batch, cls, -1)
    # Both in [N, C, K]    top K of each class, K each class
    topk_scores_all, topk_inds_all = torch.topk(heat_map, K)

    # topk_inds_all = topk_inds_all % (height * width) # todo: this seems redudant
    # [N, C, K]
    topk_ys = (topk_inds_all / width).float()
    topk_xs = (topk_inds_all % width).float()

    assert isinstance(topk_xs, torch.cuda.FloatTensor)
    assert isinstance(topk_ys, torch.cuda.FloatTensor)

    # Select topK examples across channel
    # [N, C, K] -----> [N, C*K]
    topk_scores_all = topk_scores_all.view(batch, -1)
    # Both in [N, K]
    topk_scores, topk_inds = torch.topk(topk_scores_all, K)
    topk_clses = (topk_inds / K).float()

    assert isinstance(topk_clses, torch.cuda.FloatTensor)

    # First expand it as 3 dimension
    topk_inds_all = _gather_feat(topk_inds_all.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_inds).view(batch, K)
    #      bs,50        bs,50          bs,50       bs,50    bs,50
    return topk_scores, topk_inds_all, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind):
    '''
    Select specific indexs on featuremap
    Args:
        feat: all results in 3 dimensions
        ind: positive index

    Returns:

    '''
    channel = feat.size(-1)
    ind = ind.unsqueeze(-1).expand(ind.size(0), ind.size(1), channel)
    feat = feat.gather(1, ind)

    return feat
# batch: 100
# index: bs, 100, 2
# feature_maps: bs, 9 * 6, h, w
# target_cls: bs, 100
# cls_num: 6
def select_point_of_interest(batch, index, feature_maps, target_cls, cls_num):
    '''
    Select POI(point of interest) on feature map
    Args:
        batch: batch size
        index: in point format or index format
        feature_maps: regression feature map in [N, C, H, W]

    Returns:

    '''
    w = feature_maps.shape[3]
    # bs, 100, 2
    if len(index.shape) == 3:
        index = index[:, :, 1] * w + index[:, :, 0]
    # bs, 100
    index = index.view(batch, -1)
    
    # [N, 9 * 6, H, W] -----> [N, H, W, 9 * 6]
    feature_maps = feature_maps.permute(0, 2, 3, 1).contiguous()
    channel = feature_maps.shape[-1]
    # [N, H, W, C] -----> [N, H*W, C]
    feature_maps = feature_maps.view(batch, -1, channel)
    # expand index in channels
    # bs, 100, C
    index = index.unsqueeze(-1).repeat(1, 1, channel)
    # select specific features bases on POIs
    feature_maps = feature_maps.gather(1, index.long()) # bs, 100, 9 * 6

    feature_maps = feature_maps.view(batch, feature_maps.shape[1], cls_num, -1) # bs, 100, 6, 9
    cls_index = target_cls.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,feature_maps.shape[-1]) # bs, 100, 1, 9

    feature_maps = feature_maps.gather(2, cls_index.long()).squeeze(2) # bs, 100, 9

    return feature_maps
