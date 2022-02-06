from bisect import bisect_right
import torch

from smoke import registry

# this
@registry.SCHEDULER.register('MultiStepLR')
def multi_step_lr_scheduler(cfg, optimizer, last):
  return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.DESCENT_RATE)

@registry.SCHEDULER.register('CosineAnnealingLR')
def circle_lr_scheduler(cfg, optimizer, last):
  return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                              T_max=cfg.SOLVER.COSINE_TMAX,
                                              eta_min=cfg.SOLVER.COSINE_ETAMIN,
                                              last_epoch=last)

@registry.SCHEDULER.register('CyclicLR')
def circle_lr_scheduler(cfg, optimizer, last):
  return torch.optim.lr_scheduler.CyclicLR(optimizer,
                                            base_lr=cfg.SOLVER.CYCLICLR.BASE_LR,
                                            max_lr=cfg.SOLVER.CYCLICLR.MAX_LR,
                                            step_size_up=cfg.SOLVER.CYCLICLR.STEPS_SIZE_UP,
                                            step_size_down=cfg.SOLVER.CYCLICLR.STEPS_SIZE_DOWN,
                                            mode=cfg.SOLVER.CYCLICLR.MODE,
                                            cycle_momentum=False,
                                            scale_fn=None,
                                            scale_mode=cfg.SOLVER.CYCLICLR.SCALE_MODE,
                                            last_epoch=last)

@registry.SCHEDULER.register('WarmupMultiStepLR')
def warm_multi_step_lr_scheduler(cfg, optimizer, last):
  return WarmupMultiStepLR(optimizer=optimizer,
                         milestones=cfg.SOLVER.STEPS,
                         gamma=cfg.SOLVER.DESCENT_RATE,
                         warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                         warmup_iters=cfg.SOLVER.WARMUP_ITERS, iter=last)

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=1000, iter=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of" " increasing integers. Got {}", milestones)
        self.milestones=milestones
        self.gamma=gamma
        self.warmup_factor=warmup_factor
        self.warmup_iters=warmup_iters
        self.iter=iter
        super().__init__(optimizer, iter)

    def get_lr(self):
        warmup_factor=1
        self.iter=self.last_epoch
        if self.iter < self.warmup_iters:
            alpha=float(self.iter) / self.warmup_iters
            warmup_factor=self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.iter)
            for base_lr in self.base_lrs
        ]
