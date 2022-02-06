import os
import datetime
import logging
import time
import json
import numpy

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

from tools.evaluate import evaluate
from smoke.utils.metric_logger import MetricLogger
from smoke.utils.comm import get_world_size
from smoke.utils.miscellaneous import mkdir
#from smoke.utils import comm


def updateBN(model, ratio):
    for k, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.grad.data.add_(ratio * torch.sign(m.weight.data))


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        start_iter,
):
    logger = logging.getLogger("smoke.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = cfg.SOLVER.MAX_ITERATION
    model.train()
    start_training_time = time.time()
    end = time.time()

    
    tensorboard_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'logs'))

    eval_step_period = cfg.SOLVER.EVALUATE_STEP_PERIOD

    mAP_name = ["mAP_1", "mAP_4", "mAP_5", "mAP_all"]
    
    best_mAP = [0.0, 0.0, 0.0, 0.0]
    # val_mAP[range_str][0] is a list of map_1 (BEV precision of Car)
    # ......                             map_4
    # ......                             map_5
    # ......                             map_all
    val_mAP = {}
    val_step = []
    # DEPTH_RANGES:  [[0,30],[0,15],[15,30]]                  for waymo
    #                [[0,30],[30,60],[0,15],[15,30],[0,60]]   for jdx
    for i, depth_range in enumerate(cfg.TEST.DEPTH_RANGES):
        range_str = "range[{},{}]".format(depth_range[0], depth_range[1])
        val_mAP[range_str] = [[], [], [], []]

    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "val_mAP.json")):
        with open(os.path.join(cfg.OUTPUT_DIR, "val_mAP.json")) as f:
            val_result = json.load(f)
            for i, depth_range in enumerate(cfg.TEST.DEPTH_RANGES):
                range_str = "range[{},{}]".format(depth_range[0], depth_range[1])
                val_mAP[range_str] = val_result[range_str]
            val_step = val_result["val_step"]

    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'val_result')):
        mkdir(os.path.join(cfg.OUTPUT_DIR, 'val_result'))

    for data, iteration in zip(data_loader, range(start_iter, max_iter)):
        # data: return dict(images=images, targets=targets, img_ids=img_ids)
        # targets: 
        ###################  for jdx  ###################
        #test  mode
        #       attri:  image_size=[src_width, src_height]  (origin)
        #       field:  K_src    origin K  in the annotation file
        #       trans_mat:  trans matrix from 480*640 to 120*160
        #       K:      changed K because of resize
        #train mode
        #       attri:  image_size=[640, 480]
        #       field:  invalid is set to 0!!!!!!!
        #       cls_weights (6):    cls_weight is depend on 1. cls_shot 2. dimension_reference
        #                           cls_shot bigger cls_weight smaller
        #                           dimension_reference bigger cls_weight smaller
        #       hm (6, 120, 160): gaussed heatmap
        #       reg (100, 3, 8): 3d box
        #       cls_ids (100): class id
        #       proj_p (100, 2): 2d center in (int) in 120*160 image plane
        #       dimensions (100, 3): dims
        #       locations (100, 3): 3d locations of the top(!!!!!) car
        #       rotys (100): rotys
        #       trans_mat: down sample matrix from 480*640 to 120*160 
        #       K: changed K because of augmentation
        #       reg_mask (100): 0/1
        ################### for waymo ###################
        #test mode if same with jdx
        #train mode    field + flip_mask (100): 0/1
        data_time = time.time() - end
        iteration += 1
        images = data["images"].tensors.to(device)
        #images = data["images"].tensors.to(device)
        targets = [target.to(device) for target in data["targets"]]
        
        cls_weights = torch.stack([t.get_field("cls_weights") for t in targets]).to(device)
        heatmaps = torch.stack([t.get_field("hm") for t in targets]).to(device)
        regression = torch.stack([t.get_field("reg") for t in targets]).to(device)
        cls_ids = torch.stack([t.get_field("cls_ids") for t in targets]).to(device)
        proj_points = torch.stack([t.get_field("proj_p") for t in targets]).to(device)
        dimensions = torch.stack([t.get_field("dimensions") for t in targets]).to(device)
        locations = torch.stack([t.get_field("locations") for t in targets]).to(device)
        rotys = torch.stack([t.get_field("rotys") for t in targets]).to(device)
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets]).to(device)
        K = torch.stack([t.get_field("K") for t in targets]).to(device)
        reg_mask = torch.stack([t.get_field("reg_mask") for t in targets]).to(device)
        iou_box = torch.stack([t.get_field("iou_box") for t in targets]).to(device)
        flip_mask = torch.stack([t.get_field("flip_mask") for t in targets]).to(device)
        conner_2d = torch.stack([t.get_field("conner_2d") for t in targets]).to(device)
        p_offsets = torch.stack([t.get_field("p_offsets") for t in targets]).to(device)
        """
        target_dict = torch.nn.ParameterDict({
            'cls_weights': torch.nn.Parameter(cls_weights, requires_grad=False).to(device) ,
            'heatmaps': torch.nn.Parameter(heatmaps, requires_grad=False).to(device) ,
            'regression': torch.nn.Parameter(regression, requires_grad=False).to(device) ,
            'cls_ids': torch.nn.Parameter(cls_ids, requires_grad=False).to(device) ,
            'proj_points': torch.nn.Parameter(proj_points, requires_grad=False).to(device) ,
            'dimensions': torch.nn.Parameter(dimensions, requires_grad=False).to(device) ,
            'locations': torch.nn.Parameter(locations, requires_grad=False).to(device) ,
            'rotys': torch.nn.Parameter(rotys, requires_grad=False).to(device) ,
            'trans_mat': torch.nn.Parameter(trans_mat, requires_grad=False).to(device) ,
            'K': torch.nn.Parameter(K, requires_grad=False).to(device) ,
            'reg_mask': torch.nn.Parameter(reg_mask, requires_grad=False).to(device) ,
            'iou_box': torch.nn.Parameter(iou_box, requires_grad=False).to(device) ,
            'flip_mask': torch.nn.Parameter(flip_mask, requires_grad=False).to(device) ,
            'conner_2d': torch.nn.Parameter(conner_2d, requires_grad=False).to(device)
        }).to(device)
        """
        # model:
        # train mode 
        # return losses, reg_loss_details
        # test mode
        #                                                  shifted!!       top car!!            filtered by this
        # retur torch.cat([clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores], dim=1)
        target_dict = {'cls_weights':cls_weights,  'heatmaps':heatmaps,  'regression':regression,
                       'cls_ids':cls_ids,  'proj_points':proj_points,  'dimensions':dimensions,
                       'locations':locations,  'rotys':rotys,  'trans_mat':trans_mat,
                       'K':K,  'reg_mask':reg_mask,  'iou_box':iou_box, 'flip_mask':flip_mask,'conner_2d':conner_2d,'p_offsets':p_offsets}
        loss_dict, losses_reg_details_dict = model(images, target_dict)
        
        for k,v in loss_dict.items():
            loss_dict[k] = torch.mean(v)
        for k,v in losses_reg_details_dict.items():
            losses_reg_details_dict[k] = torch.mean(v)
        #loss_dict, losses_reg_details_dict = model(images.tensors, cls_weights=cls_weights, heatmaps=heatmaps, 
        #                                           regression=regression, cls_ids=cls_ids, proj_points=proj_points, 
        #                                           dimensions=dimensions, locations=locations, rotys=rotys, 
        #                                          trans_mat=trans_mat, K=K, reg_mask=reg_mask, iou_box=iou_box, 
        #                                           flip_mask=flip_mask, conner_2d=conner_2d)
        losses = sum(loss for loss in loss_dict.values())
        # yao add
        if(iteration>1000 and torch.gt(losses,100)):
            continue
            
        ## reduce losses over all GPUs for logging purposes
        #loss_dict_reduced = reduce_loss_dict(loss_dict)
        #losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        #meters.update(loss=losses_reduced, **loss_dict_reduced)
        meters.update(loss=losses, **loss_dict)

        ### lyb:added
        #losses_reg_details_dict_reduced = reduce_loss_dict(losses_reg_details_dict)
        #meters.update(**losses_reg_details_dict_reduced)
        meters.update(**losses_reg_details_dict)

        optimizer.zero_grad()
        losses.backward()
        # false
        if cfg.PRUNNING.BN_SPARSITY:
            updateBN(model, cfg.PRUNNING.BN_RATIO)
        #print("***********************************************\n\n")
        ###for name,p in model.named_parameters():
            #if(p.grad is not None):
            #    print(name, p.grad.norm(), p.requires_grad)
            ###if(p.grad is not None):
                ###if (torch.gt(p.grad.norm(), 5)):
                    ###torch.nn.utils.clip_grad_norm_(p, max_norm=1, norm_type=2)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        
        
        #logger.info("***********************************************\n\n")
        #logger.info(f'{losses}')
        #for name,p in model.named_parameters():
        #    if(p.grad is not None):
        #        logger.info(f'{name}, {p.grad.norm()}, {p.requires_grad}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 10 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f}",
                        "max men: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    # median of window_size(20), ( global average )
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                )
            )

        # Evalution of each epoch: save tensorboard, eval results, checkpoints
        if iteration % eval_step_period == 0 or iteration == max_iter:
            model.eval()
            # result_dict = {}
            # result_dict[dataset_name] = result_dict
            #       result_dict = {}
            #
            #       result_dict[range_str]["APbbox"] = 2D box precision
            #       result_dict[range_str]["APbev"] =  BEV precision
            #       result_dict[range_str]["AP3d"] =   3D box precision
            #       result_dict[range_str]["APaos"] =  2D box orientation
            #       result_dict[range_str]["ARbbox"] = 2D box recall
            #       result_dict[range_str]["ARbev"] =  BEV recall
            #       result_dict[range_str]["AR3d"] =   3D box recall
            #       result_dict[range_str]["bev_mAP"] = [map_1, map_4, map_5, map_all]
            #                   map_1:  BEV precision of Car
            #                   map_4:  BEV precision of Car, Cyclist, Pedestrian, Truck
            #                   map_5:  BEV precision of Car, Cyclist, Pedestrian, Truck, Bus
            #                 map_all:  BEV precision of all
            #
            #       result_dict["class_id_name"] = {0:Car, ...}
            #       result_dict[range_str]["print"] = ...
            result_dict = evaluate(cfg, model)
            if result_dict is not None and len(result_dict) != 0:
                ### save multi range result
                for i, depth_range in enumerate(cfg.TEST.DEPTH_RANGES):
                    range_str = "range[{},{}]".format(depth_range[0], depth_range[1])
                    # only the first test datasets?
                    map_temp = result_dict[cfg.DATASETS.TEST[0]][range_str]["bev_mAP"]
                    # {car:bev precision, ......, for each class}
                    # +
                    # {mAP(Car,Pedestrian,Cyclist,Truck) : map_1}
                    # +
                    # {mAP(Car,Pedestrian,Cyclist,Truck,Bus) : map_5}
                    # +
                    # {mAP(all) : map_all}
                    ap_dict = {}
                    for id, v in enumerate(result_dict[cfg.DATASETS.TEST[0]][range_str]["APbev"]):
                        ap_dict[result_dict[cfg.DATASETS.TEST[0]]["class_id_name"][id]] = v
                    ap_dict["mAP(Car,Pedestrian,Cyclist,Truck)"] = map_temp[1]
                    ap_dict["mAP(Car,Pedestrian,Cyclist,Truck,Bus)"] = map_temp[2]
                    ap_dict["mAP(all)"] = map_temp[3]

                    bev_mAP_str = "\nbev_{}:\n".format(range_str)
                    # [4]
                    for k in range(len(best_mAP)):
                        val_mAP[range_str][k].append(map_temp[k])
                        bev_mAP_str += "{}: {}    ".format(mAP_name[k], round(map_temp[k], 2))
                    bev_mAP_str += "\n"

                    ### first depth_range is default
                    if i == 0:
                        checkpointer.save(
                            "model_iter_{:07d}_{}_{}_{}_{}".format(iteration, round(map_temp[0], 2),
                                                                   round(map_temp[1], 2), round(map_temp[2], 2),
                                                                   round(map_temp[3], 2)),
                            iteration)
                        for j in range(len(best_mAP)):
                            if val_mAP[range_str][j][-1] > best_mAP[j]:
                                best_mAP[j] = val_mAP[range_str][j][-1]
                                checkpointer.save("{}_best_model_iter_{:07d}_{}".format(mAP_name[j], iteration,
                                                                                        round(best_mAP[j], 2)),
                                                  iteration)

                                with open(os.path.join(cfg.OUTPUT_DIR, 'val_result/{}_best.txt'.format(mAP_name[j])),
                                          "w") as f:
                                    for test_set in cfg.DATASETS.TEST:
                                        f.write("Evaluation result of dataset [{}]:\n".format(test_set))
                                        f.write(result_dict[test_set]["print"])

                    tensorboard_writer.add_scalars('bev_AP_' + range_str, ap_dict, iteration)

                val_step.append(iteration)
                with open(os.path.join(cfg.OUTPUT_DIR, "val_mAP.json"), 'w') as file_object:
                    dict_save = {"val_step": val_step}
                    for i, depth_range in enumerate(cfg.TEST.DEPTH_RANGES):
                        range_str = "range[{},{}]".format(depth_range[0], depth_range[1])
                        dict_save[range_str] = val_mAP[range_str]
                    json.dump(dict_save, file_object)

                #tensorboard_writer.add_scalars('loss', {'cls_loss': loss_dict["cls_loss"].data,
                #                                        'reg_loss': loss_dict["reg_loss"].data,
                #                                        'reg_loss_loc': losses_reg_details_dict["reg_loss_loc"].data,
                #                                        'reg_loss_ori': losses_reg_details_dict["reg_loss_ori"].data,
                #                                        'reg_loss_dim': losses_reg_details_dict["reg_loss_dim"].data},
                #                               iteration)
                #tensorboard_writer.add_scalar("learning rate", optimizer.param_groups[0]["lr"], iteration)

                with open(os.path.join(cfg.OUTPUT_DIR, 'val_result/result_iter_{:07d}.txt'.format(iteration)),
                          "w") as f:
                    for test_set in cfg.DATASETS.TEST:
                        f.write("Evaluation result of dataset [{}]:\n".format(test_set))
                        f.write(result_dict[test_set]["print"])
                        logger.info("Evaluation result of dataset [{}]:".format(test_set))
                        logger.info("\n" + result_dict[test_set]["print"])

            model.train()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / (max_iter)))
