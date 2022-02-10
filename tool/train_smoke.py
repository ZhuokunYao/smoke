import argparse
import os
import torch
#from smoke.utils import comm
from smoke.utils.miscellaneous import mkdir
from smoke.utils.logger import setup_logger
from smoke.utils.collect_env import collect_env_info
from smoke.utils.envs import seed_all_rng
from smoke.utils.check_point import Checkpointer
from smoke.config import cfg
from smoke.data import make_data_loader
from smoke.solver import make_optimizer, make_lr_scheduler
#from smoke.engine import launch
from smoke.engine.trainer import do_train
from smoke.modeling.detector import build_detection_model

from fvcore.common.checkpoint import Checkpointer as fv_Checkpointer

def default_argument_parser():
    parser = argparse.ArgumentParser(description="Monocular 3D Object Detection Training")
    """
    config_file is used to set config args
    others is used to DDP setting
    """
    parser.add_argument("--config_file", default="configs/smoke_jdx_resnet18_640x480.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus *per machine*")
    #parser.add_argument("--num_machines", type=int, default=1)
    #parser.add_argument("--machine_rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    return parser


def default_setup(cfg, args):
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    args.num_gpus = torch.cuda.device_count()
    cfg.DATALOADER.NUM_WORKERS *= args.num_gpus
    cfg.SOLVER.IMS_PER_BATCH *= args.num_gpus
    #rank = comm.get_rank()
    #logger = setup_logger(output_dir, rank)
    #rank = comm.get_rank()
    logger = setup_logger(output_dir, 0)
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info("Collecting environment info")
    logger.info("\n" + collect_env_info())
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

def train(cfg, model, device, checkpointer):
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    checkpointer.init(model, optimizer, scheduler)
    start_iter = 0

    fv_Checkpointer(model).load('pretrain/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped.pth')
    """
    if cfg.SOLVER.PRETRAIN_MODEL and cfg.SOLVER.PRETRAIN_MODEL.startswith('pretrain'):
        fv_Checkpointer(model).load(cfg.SOLVER.PRETRAIN_MODEL)
    elif cfg.SOLVER.PRETRAIN_MODEL:
        if cfg.SOLVER.PRETRAIN_MODEL:
            checkpointer.load_model()
        if cfg.SOLVER.RESUME:
            checkpointer.load_optimizer()
            # If want to finetune and load saved scheduler prarms, please uncomment it.
            checkpointer.load_scheduler()
            start_iter = checkpointer.load_param('iteration')
    """

    data_loader = make_data_loader(cfg, is_train=True)

    do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        start_iter
    )

def main(args):
    cfg.merge_from_file(args.config_file)
    default_setup(cfg, args)
    checkpointer = Checkpointer(cfg.SOLVER.PRETRAIN_MODEL, cfg.SOLVER.EXCLUDE_LAYERS, cfg.OUTPUT_DIR)
    # [] if PRETRAIN_MDOEL = ''
    cfg.MODEL.BACKBONE.CHANNELS = checkpointer.load_param('backbone_channels')
    if len(cfg.MODEL.BACKBONE.CHANNELS) > 0:
        cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS = cfg.MODEL.BACKBONE.CHANNELS[-1]
    #train :return losses, reg_loss_details   2 dicts
    #test  :torch.cat([clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores], dim=1)
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model = torch.nn.DataParallel(model)
    model.to(device)
    #distributed = comm.get_world_size() > 1
    #if distributed:
    #    model = torch.nn.parallel.DistributedDataParallel(
    #        model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
    #        find_unused_parameters=True)

    train(cfg, model, device, checkpointer)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    # pytorch DDP
    main(args)
    """
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    """
