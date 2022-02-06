# from .build import make_optimizer
from smoke import registry
from .lr_optimizer import sgd_optimizer, adam_optimizer, rmsprop_optimizer
from .lr_scheduler import multi_step_lr_scheduler, circle_lr_scheduler, warm_multi_step_lr_scheduler

def make_optimizer(cfg, model):
  return registry.OPTIMIZER[cfg.SOLVER.OPTIMIZER](cfg, model)

def make_lr_scheduler(cfg, optimizer, last=-1):
    return registry.SCHEDULER[cfg.SOLVER.SCHEDULER](cfg, optimizer, last=last)
