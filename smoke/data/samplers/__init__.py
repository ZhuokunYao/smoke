from smoke import registry
from .grouped_batch_sampler import GroupedBatchSampler
from .distributed_sampler import (
    RepeatFactorTrainingSampler,
    TrainingSampler,
    InferenceSampler,
)

__all__ = ["GroupedBatchSampler",
           "TrainingSampler",
           "InferenceSampler"]

def build_dataloder_sampler(cfg, dataset):
    #                       "TrainingSampler"
    return registry.SAMPLER[cfg.DATALOADER.SAMPLER.TYPE](cfg, dataset)
