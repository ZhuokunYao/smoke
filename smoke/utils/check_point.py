import logging
import os

import torch

from smoke.utils.model_serialization import load_state_dict
from smoke.utils.imports import import_file
from smoke.utils.model_zoo import cache_url

class Checkpointer():
    def __init__(
            self,
            ckpt_file,  # ''   PRETRAIN_MODEL
            exclude_layers,  # ''
            save_dir="",  #  "./checkpoint/jdx_resnet18_640x480_crop_resize"
            logger=None,
    ):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.save_dir = save_dir
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        if not os.path.exists(ckpt_file):
            self.logger.info("No checkpoint found. Initializing model from scratch")
            self.ckpt_file = None
            self.checkpoint = None
            self.exclude_layers = None
        else:
            self.ckpt_file = ckpt_file
            self.checkpoint = self.load_file()
            self.exclude_layers = exclude_layers

    # 3 things to save and load
    def init(self, model=None, optimizer=None, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save(self, name, iteration):
        if not self.save_dir:
            return
        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data['iteration'] = iteration
        data['backbone_channels'] = self.checkpoint['backbone_channels'] if self.checkpoint is not None else []

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load_optimizer(self):
        self.logger.info("Loading optimizer from {}".format(self.ckpt_file))
        self.optimizer.load_state_dict(self.checkpoint["optimizer"])

    def load_scheduler(self):
        if "scheduler" in self.checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(self.ckpt_file))
            self.scheduler.load_state_dict(self.checkpoint["scheduler"])
    
    # return: model, optimizer, scheduler, iteration, backbone_channels
    def load_file(self):
        self.logger.info("Loading checkpoint from {}".format(self.ckpt_file))
        return torch.load(self.ckpt_file, map_location=torch.device("cpu"))

    def load_model(self):
        load_state_dict(self.model, self.checkpoint["model"], self.exclude_layers)
    
    # load a param
    def load_param(self, param_key):
        if self.checkpoint is not None and param_key in self.checkpoint.keys():
            return self.checkpoint[param_key]
        else:
            return []
