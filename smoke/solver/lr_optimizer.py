import torch
from smoke import registry

@registry.OPTIMIZER.register('SGD')
def sgd_optimizer(cfg, model):
  return torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                         weight_decay=cfg.SOLVER.WEIGHT_DECAY, nesterov=False)

@registry.OPTIMIZER.register('NesterovSGD')
def sgd_nesterov_optimizer(cfg, model):
  return torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                         weight_decay=cfg.SOLVER.WEIGHT_DECAY, nesterov=True)


@registry.OPTIMIZER.register('Adam')
def adam_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        # 8e-4
        lr = cfg.SOLVER.BASE_LR
        if "bias" in key:
                                      # 2
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
        params += [{"params": [value], "lr": lr}]
    return torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR)


@registry.OPTIMIZER.register('RMSProp')
def rmsprop_optimizer(cfg, model):
  return torch.optim.RMSProp(model.parameters(), lr=cfg.SOLVER.BASE_LR, alpha=cfg.SOLVER.ALPHA)
