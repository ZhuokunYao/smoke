import numpy as np
import argparse
import os

import torch
import torch.nn as nn

from smoke.modeling.detector import build_detection_model
from smoke.config import cfg
from smoke.utils.model_serialization import load_state_dict

def arg_parse():
    parser = argparse.ArgumentParser(description="SMOKE Prunning script")
    parser.add_argument("--config-file", type=str,
        default="configs/smoke_jdx_resnet18_640x480.yaml")
    parser.add_argument("--ckpt", type=str,
        default="path/to/ur/checkpoint.pth")
    parser.add_argument("--save_dir", type = str,
        default = "pretrained_model/")
    parser.add_argument('--percent', type=float, help='the ratio of prunning',
        default=0.5)
    parser.add_argument('--prune_header', type=bool, help='Whether to prune the header layers',
                        default=False)
    parser.add_argument('--prune_skip', type=bool, help='Whether to prune the header layers',
                        default=True)
    return parser.parse_args()



reg_header_id = ['heads.predictor.regression_head.0',
                 'heads.predictor.regression_head.1',
                  'heads.predictor.regression_head.3']
cls_header_id = ['heads.predictor.class_head.0',
                  'heads.predictor.class_head.1',
                  'heads.predictor.class_head.3']

# The last conv ID of 2,3,4 stage, which should have the same shape as the
# downsample opration in skip connection

remained_id = [3, 12]

downsample_layer_id = [28, 44, 60]

stage_layer_id = [18, 34, 50, 66]

predictor_layer = ['heads.predictor.class_head.3', 'heads.predictor.regression_head.3']

 # print the layers name of the network
def select_pruned_layers(model):
    cls_head_features = []
    reg_head_features = []
    backbone_features = []
    for id, [name, layer] in enumerate(list(model.named_modules())):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d)or isinstance(layer, nn.BatchNorm2d):
            print(id, name, layer)
            if 'class_head' in name:
                cls_head_features.append(id)
            if 'regression_head' in name:
                reg_head_features.append(id)
            if 'backbone' in name:
                backbone_features.append(id)
    return backbone_features, cls_head_features, reg_head_features

def calc_total_bn(model, features):
    # calculate the number of the weights of bn layers
    total = 0
    total_k_1 = 0
    total_other = 0
    modules = list(model.named_modules())
    for i, id in enumerate(features):
        name, layer = modules[id]
        if isinstance(layer, nn.Conv2d):
            if id in remained_id or id in downsample_layer_id  or id in stage_layer_id:
                continue

            name_bn, layer_bn = modules[features[i + 1]]
            if isinstance(layer_bn, nn.BatchNorm2d):
                if layer.kernel_size == (1, 1):
                    total_k_1 += layer_bn.weight.data.shape[0]
                else:
                    total_other += layer_bn.weight.data.shape[0]
                total += layer_bn.weight.data.shape[0]
    print("The number of the weights of pruned BN layers: {} ".format(total))
    print("The number of the weights of pruned BN layers after conv_1x1: {} ".format(total_k_1))
    print("The number of the weights of pruned BN layers after other convs: {} ".format(total_other))
    return total, total_k_1, total_other

def get_threshold(model, features):
    # sort the list of BN weights and get the threshold
    modules = list(model.named_modules())
    total, total_k_1, total_other = calc_total_bn(model, features)
    bn = torch.zeros(total)
    bn_k_1 = torch.zeros(total_k_1)
    bn_other = torch.zeros(total_other)
    index_k_1 = 0
    index_other = 0
    index = 0
    for i, id in enumerate(features):
        name, layer = modules[id]
        if isinstance(layer, nn.Conv2d):
            if id in remained_id or id in downsample_layer_id or id in stage_layer_id:
                continue

            name_bn, layer_bn = modules[features[i + 1]]
            print(name_bn)
            if isinstance(layer_bn, nn.BatchNorm2d):
                size = layer_bn.weight.data.shape[0]
                bn[index:(index + size)] = layer_bn.weight.data.abs().clone()
                index += size
                if layer.kernel_size == (1, 1):
                    size_k_1 = layer_bn.weight.data.shape[0]
                    bn_k_1[index_k_1:(index_k_1 + size_k_1)] = layer_bn.weight.data.abs().clone()
                    index_k_1 += size_k_1
                else:
                    size_other = layer_bn.weight.data.shape[0]
                    bn_other[index_other:(index_other + size_other)] = layer_bn.weight.data.abs().clone()
                    index_other += size_other

    y, i = torch.sort(bn, descending = True)
    thre_index = int(total * (1-args.percent))
    thre = y[thre_index].cuda()
    print("Percentage: {:.2f}, Threshold: {:.4f}, Index: {:d}".format(
            args.percent, thre, thre_index))

    # y_k_1, i_k_1 = torch.sort(bn_k_1)
    # thre_index_k_1 = int(total_k_1 * args.percent)
    # thre_k_1 = y_k_1[thre_index_k_1].cuda()
    # print("Percentage conv_1x1: {:.2f}, Threshold: {:.4f}, Index: {:d}".format(
    #     args.percent, thre_k_1, thre_index_k_1))

    # y_other, i_other = torch.sort(bn_other)
    # thre_index_other = int(total_other * args.percent)
    # thre_other = y_other[thre_index_other].cuda()
    # print("Percentage other: {:.2f}, Threshold: {:.4f}, Index: {:d}".format(
    #     args.percent, thre_other, thre_index_other))
    return thre

def calc_pruned_mask_for_layer(modules, features, idx, thre, pruned = True):
    name, layer = modules[features[idx]]
    name_bn, layer_bn = modules[features[idx + 1]]
    weight_copy = layer_bn.weight.data.abs().clone()
    if layer.kernel_size == (1, 1) or not pruned:
        mask = torch.ones(weight_copy.shape).cuda()
    else:
        mask = weight_copy.gt(thre).float().cuda()
    # pruned = mask.shape[0] - torch.sum(mask)
    layer_bn.weight.data.mul_(mask)
    layer_bn.bias.data.mul_(mask)
    print('LayerID: {:d}--{:s} \t TotalChannel: {:d} \t RemainingChannel: {:d}'.
          format(features[idx], modules[features[idx]][0], mask.shape[0], int(torch.sum(mask))))
    return int(torch.sum(mask)), mask

# calculate the backbone mask for all layers, but remain the structure of
# the output stage for boxheader
def calc_backbone_mask(model, backbone_features, thre):
    # the pruned and the remaining number of BN channels
    modules = list(model.named_modules())
    pruned = 0
    backbone_channel = [3]
    backbone_mask = [torch.ones(3)]
    print('--' * 30)
    print("Process and calculate the mask of the backbone list")
    # i = 0
    # while i < len(backbone_features):
    for index, id in enumerate(backbone_features):
        name, layer = modules[id]
        # if the previous and next Conv2d is pointwise layer,
        # use the mask of BN followed by the depthwise layer
        print(name, layer)
        if isinstance(layer, nn.Conv2d):
            if id in remained_id:
                name_bn, layer_bn = modules[backbone_features[index + 1]]
                dontp = layer_bn.weight.data.numel()
                mask = torch.ones(layer_bn.weight.data.shape)
                print(
                    'LayerID: {:d}--{:s} \t TotalChannel: {:d} \t RemainingChannel: {:d}'.format(
                        id, name, dontp, int(dontp)))
                backbone_channel.append(int(dontp))
                backbone_mask.append(mask.clone())
            elif id in stage_layer_id:
                if args.prune_skip and id != stage_layer_id[0]:
                    # The output of the stage should be the same as the downsample in pre-block
                    name_bn, layer_bn = modules[backbone_features[index + 1]]
                    dontp = layer_bn.weight.data.numel()
                    ch, mask = backbone_channel[-2], backbone_mask[-2]
                    print(
                        'LayerID: {:d}--{:s} \t TotalChannel: {:d} \t RemainingChannel: {:d}'.format(
                            id, name, dontp, ch))
                    backbone_channel.append(ch)
                    backbone_mask.append(mask)
                else:
                    name_bn, layer_bn = modules[backbone_features[index + 1]]
                    dontp = layer_bn.weight.data.numel()
                    mask = torch.ones(layer_bn.weight.data.shape)
                    print(
                        'LayerID: {:d}--{:s} \t TotalChannel: {:d} \t RemainingChannel: {:d}'.format(
                            id, name, dontp, int(dontp)))
                    backbone_channel.append(int(dontp))
                    backbone_mask.append(mask.clone())
            elif id in downsample_layer_id:
                if args.prune_skip:
                    # prun the skip layer based on the previous layer
                    name_bn, layer_bn = modules[backbone_features[index + 1]]
                    dontp = layer_bn.weight.data.numel()
                    ch, mask = backbone_channel[-1], backbone_mask[-1]
                    print(
                        'LayerID: {:d}--{:s} \t TotalChannel: {:d} \t RemainingChannel: {:d}'.format(
                            id, name, dontp, ch))
                    backbone_channel.append(ch)
                    backbone_mask.append(mask)
                else:
                    name_bn, layer_bn = modules[backbone_features[index + 1]]
                    dontp = layer_bn.weight.data.numel()
                    mask = torch.ones(layer_bn.weight.data.shape)
                    print(
                        'LayerID: {:d}--{:s} \t TotalChannel: {:d} \t RemainingChannel: {:d}'.format(
                            id, name, dontp, int(dontp)))
                    backbone_channel.append(int(dontp))
                    backbone_mask.append(mask.clone())
            else:
                # The next layer is the skip layer
                if index + 2 < len(backbone_features):
                    name_skip, _ = modules[backbone_features[index + 2]]
                if 'downsample' in name_skip:
                    if args.prune_skip:
                        ch, mask = calc_pruned_mask_for_layer(modules, backbone_features, index, thre, pruned=True)
                        pruned = pruned + (mask.shape[0] - ch)
                        backbone_channel.append(ch)
                        backbone_mask.append(mask)
                    else:
                        name_bn, layer_bn = modules[backbone_features[index + 1]]
                        dontp = layer_bn.weight.data.numel()
                        mask = torch.ones(layer_bn.weight.data.shape)
                        print(
                            'LayerID: {:d}--{:s} \t TotalChannel: {:d} \t RemainingChannel: {:d}'.format(
                                id, name, dontp, int(dontp)))
                        backbone_channel.append(int(dontp))
                        backbone_mask.append(mask.clone())
                else:
                    ch, mask = calc_pruned_mask_for_layer(modules, backbone_features, index, thre, pruned = True)
                    pruned = pruned + (mask.shape[0] - ch)
                    backbone_channel.append(ch)
                    backbone_mask.append(mask)
        elif isinstance(layer, nn.ConvTranspose2d):
            if id in remained_id:
                name_bn, layer_bn = modules[backbone_features[index + 1]]
                dontp = layer_bn.weight.data.numel()
                mask = torch.ones(layer_bn.weight.data.shape)
                print(
                    'LayerID: {:d}--{:s} \t TotalChannel: {:d} \t RemainingChannel: {:d}'.format(
                        id, name, dontp, int(dontp)))
                backbone_channel.append(int(dontp))
                backbone_mask.append(mask.clone())
            else:
                ch, mask = calc_pruned_mask_for_layer(modules, backbone_features, index, thre, pruned = True)
                pruned = pruned + (mask.shape[0] - ch)
                backbone_channel.append(ch)
                backbone_mask.append(mask)

    print('--' * 30)
    print(backbone_channel)
    return backbone_channel, backbone_mask

def get_backbone_mask(idx):
    return backbone_mask[idx]

def get_backbone_channel(idx):
    return backbone_channel[idx]

def calc_head_mask(model, head_features, thre):
    # the pruned and the remaining number of BN channels
    modules = list(model.named_modules())
    pruned = 0
    head_channel = [get_backbone_channel(-1)]
    head_mask = [get_backbone_mask(-1)]
    print('--' * 30)
    print("Process and calculate the mask of the head list")
    for index, id in enumerate(head_features):
        name, layer = modules[id]
        print(name, layer)
        if isinstance(layer, nn.Conv2d):
            # The predictor layer, not followed the BN layer
            if name in predictor_layer:
                mask = torch.ones(layer.weight.data.shape[0]).cuda()
                ch = layer.weight.data.shape[0]
                print('LayerID: {:d}--{:s} \t TotalChannel: {:d} \t RemainingChannel: {:d}'.format(id, name, ch, ch))
                head_channel.append(ch)
                head_mask.append(mask.clone())
            else:
                ch, mask = calc_pruned_mask_for_layer(modules, head_features, index, thre, pruned=args.prune_header)
                pruned = pruned + (mask.shape[0] - ch)
                head_channel.append(ch)
                head_mask.append(mask)
    return head_channel, head_mask

def prune_bn(name, layer_0, layer_1, start_mask):
    print('BatchNorm2d >>>>>>>>>>>>> ', name)
    print(layer_0)
    index_1 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
    if index_1.size == 1:
        index_1 = np.resize(index_1, (1,))
    # process the BN layer
    layer_1.weight.data = layer_0.weight.data[index_1.tolist()].clone()
    layer_1.bias.data = layer_0.bias.data[index_1.tolist()].clone()
    layer_1.running_mean = layer_0.running_mean[index_1.tolist()].clone()
    layer_1.running_var = layer_0.running_var[index_1.tolist()].clone()
    layer_1.num_features = index_1.size
    print('Src shape: {:d}, Dst shape {:d}.'.format(
        layer_0.num_features, layer_1.num_features))

def prune_conv(name, layer_0, layer_1, start_mask, end_mask):
    print('Conv2d >>>>>>>>>>>>> ', name)
    print(layer_0)
    index_0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
    index_1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
    if index_0.size == 1:
        index_0 = np.resize(index_0, (1,))
    if index_1.size == 1:
        index_1 = np.resize(index_1, (1,))

    weight = layer_0.weight.data.clone()

    # for pointwise convolution
    if layer_0.kernel_size == (1, 1):
        weight = weight[index_1.tolist(), :, :, :].clone()
        weight = weight[:, index_0.tolist(), :, :].clone()
        layer_1.weight.data = weight.clone()
        if layer_0.bias is not None:
            layer_1.bias.data = layer_0.bias.data[index_1.tolist()].clone()
    else:
        # for general convolution
        if layer_0.groups == 1:
            weight = weight[:, index_0.tolist(), :, :].clone()
            weight = weight[index_1.tolist(), :, :, :].clone()
            layer_1.weight.data = weight.clone()
            if layer_0.bias is not None:
                layer_1.bias.data = layer_0.bias.data[index_1.tolist()].clone()
        # for depthwise convolution
        elif layer_0.groups > 1:
            weight = weight[index_0.tolist(), :, :, :].clone()
            layer_1.weight.data = weight.clone()
            if layer_0.bias is not None:
                layer_1.bias.data = layer_0.bias.data[index_0.tolist()].clone()
    print('Src shape: [{:d},{:d},{:d},{:d}], Dst shape [{:d},{:d},{:d},{:d}].'.format(
        layer_0.weight.data.shape[0], layer_0.weight.data.shape[1], layer_0.weight.data.shape[2],
        layer_0.weight.data.shape[3],
        layer_1.weight.data.shape[0], layer_1.weight.data.shape[1], layer_1.weight.data.shape[2],
        layer_1.weight.data.shape[3], ))

def prune_deconv(name, layer_0, layer_1, start_mask, end_mask):
    print('ConvTranspose2d >>>>>>>>>>>>> ', name)
    print(layer_0)
    index_0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
    index_1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
    if index_0.size == 1:
        index_0 = np.resize(index_0, (1,))
    if index_1.size == 1:
        index_1 = np.resize(index_1, (1,))

    weight = layer_0.weight.data.clone()

    # for general convolution
    weight = weight[index_0.tolist(), :, :, :].clone()
    weight = weight[:, index_1.tolist(), :, :].clone()
    layer_1.weight.data = weight.clone()
    if layer_0.bias is not None:
        layer_1.bias.data = layer_0.bias.data[index_1.tolist()].clone()

    print('Src shape: [{:d},{:d},{:d},{:d}], Dst shape [{:d},{:d},{:d},{:d}].'.format(
        layer_0.weight.data.shape[0], layer_0.weight.data.shape[1], layer_0.weight.data.shape[2],
        layer_0.weight.data.shape[3],
        layer_1.weight.data.shape[0], layer_1.weight.data.shape[1], layer_1.weight.data.shape[2],
        layer_1.weight.data.shape[3], ))

def prune_backbone(old_modules, new_modules, backbone_features):
    backbone_channel, backbone_mask = calc_backbone_mask(model, backbone_features, thre)
    start_mask = backbone_mask[0]
    end_mask = backbone_mask[1]
    pre_stage_out_mask = backbone_mask[0]
    layer_id_in_cfg = 1

    print('--' * 30)
    print("Begin pruning the backbone...")
    total = 0
    idx = 0
    while idx < len(backbone_features):
        id = backbone_features[idx]
        name, layer_0 = old_modules[id]
        _, layer_1 = new_modules[id]
        print(name)

        # special process for skip connection
        if id in downsample_layer_id:
            if isinstance(layer_0, nn.BatchNorm2d):
                prune_bn(name, layer_0, layer_1, start_mask)
            elif isinstance(layer_0, nn.Conv2d):
                prune_conv(name, layer_0, layer_1, pre_stage_out_mask, end_mask)
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(backbone_mask):
                    end_mask = backbone_mask[layer_id_in_cfg]
                print("start_mask", start_mask.shape)
                print("end_mask", end_mask.shape)
        else:
            if isinstance(layer_0, nn.BatchNorm2d):
                prune_bn(name, layer_0, layer_1, start_mask)
            elif isinstance(layer_0, nn.Conv2d):
                prune_conv(name, layer_0, layer_1, start_mask, end_mask)
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(backbone_mask):
                    end_mask = backbone_mask[layer_id_in_cfg]
                print("start_mask", start_mask.shape)
                print("end_mask", end_mask.shape)
                # store the output mask of last convlution in one stage
            elif isinstance(layer_0, nn.ConvTranspose2d):
                prune_deconv(name, layer_0, layer_1, start_mask, end_mask)
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(backbone_mask):
                    end_mask = backbone_mask[layer_id_in_cfg]
                print("start_mask", start_mask.shape)
                print("end_mask", end_mask.shape)
                # store the output mask of last convlution in one stage

            if id in stage_layer_id:
                pre_stage_out_mask = start_mask

        idx += 1
    return backbone_channel, backbone_mask
    print('--' * 30)

def prune_head(old_modules, new_modules, head_features):
    head_channel, head_mask = calc_head_mask(model, head_features, thre)

    print("Begin to prune the head...")
    layer_id_in_cfg = 1
    start_mask = head_mask[0]
    end_mask = head_mask[1]

    for id in head_features:
        name, layer_0 = old_modules[id]
        _, layer_1 = new_modules[id]
        print(name)
        if isinstance(layer_0, nn.BatchNorm2d):
            prune_bn(name, layer_0, layer_1, start_mask)
        elif isinstance(layer_0, nn.Conv2d):
            prune_conv(name, layer_0, layer_1, start_mask, end_mask)
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(head_mask):
                end_mask = head_mask[layer_id_in_cfg]
            print("start_mask", start_mask.shape)
            print("end_mask", end_mask.shape)
    return head_channel, head_mask

if __name__ == '__main__':
    args = arg_parse()
    CUDA = torch.cuda.is_available()
    print("Loading network ...")
    cfg.merge_from_file(args.config_file)
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_detection_model(cfg).cuda()
    checkpoint = torch.load(args.ckpt)
    load_state_dict(model, checkpoint['model'])
    old_modules = list(model.named_modules())
    new_model = build_detection_model(cfg).cuda()
    new_modules = list(new_model.named_modules())

    # set the layers to prune
    backbone_features, cls_head_features, reg_head_features = select_pruned_layers(model)

    thre = get_threshold(model, backbone_features)

    backbone_channel, backbone_mask = prune_backbone(old_modules, new_modules, backbone_features)

    cls_head_channel, cls_head_mask = prune_head(old_modules, new_modules, cls_head_features)

    reg_head_channel, reg_head_mask = prune_head(old_modules, new_modules, reg_head_features)

    print([backbone_channel, cls_head_channel, reg_head_channel])
    num_parameters1 = sum([param.nelement() for param in model.parameters()])
    num_parameters2 = sum([param.nelement() for param in new_model.parameters()])
    print("Parameters of original model: \n" + str(num_parameters1) + "\n")
    print("Parameters of pruned model: \n" + str(num_parameters2) + "\n")
    print("Percentage of prunning: %.2f\n" % (1 - float(num_parameters2) / num_parameters1))

    # save the new pruned model
    pruned_weight = (args.ckpt.split('/')[-1])[:-4] + "_p{}.pth".format(int(args.percent*10))
    print('Save weight file %s' % os.path.join(args.save_dir, pruned_weight))
    torch.save({'backbone_channels': backbone_channel,  'model': new_model.state_dict()},
               os.path.join(args.save_dir, pruned_weight))
    print('Done!')
