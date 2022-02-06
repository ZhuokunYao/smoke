import numpy as np
import torch.utils.data
import logging

from smoke.utils.comm import get_world_size
from smoke.utils.imports import import_file
from smoke.utils.envs import seed_all_rng

from . import datasets as D
from .samplers import build_dataloder_sampler, InferenceSampler
from .transforms import build_transforms
from .kitti_utils import calculate_class_weight
from .collate_batch import BatchCollator


def build_dataset(cfg, transforms, dataset_catalog, is_train=True):
    '''
    Args:
        dataset_list (list[str]): Contains the names of the datasets.
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing

    Returns:

    '''
    
    #["jdx_fusion_front", "jdx2020_tracker", "jdx2021_tracker", "jdx2021_fusion"]
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    # ["jdx_fusion_front", "jdx2020_tracker", "jdx2021_tracker", "jdx2021_fusion"]
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(
                dataset_list))

    datasets = []
    for dataset_name in dataset_list:
        #get:  factory:JDXDataset,   args:  root:datasets/jdx2020_tracker/training/,split:train
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]

        args["cfg"] = cfg
        args["is_train"] = is_train
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets
    else:
        # for training, calculate the weights for all datasets, concatenate all datasets into a single one
        logger = logging.getLogger("smoke.train")
        num_classes = len(cfg.DATASETS.DETECT_CLASSES)
        cls_shots = np.zeros([num_classes], dtype=np.int32)
        cls_weights = np.ones([num_classes], dtype=np.float32)
        for i in range(len(datasets)):
            cls_shots += datasets[i].cls_shots
        cls_weights = calculate_class_weight(cls_shots,
            [cfg.MODEL.SMOKE_HEAD.CLS_REWEIGHT_SHOT_GAMMA,
            cfg.MODEL.SMOKE_HEAD.CLS_REWEIGHT_SIZE_GAMMA],
            cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE)
        for i in range(len(datasets)):
            datasets[i].cls_weights = cls_weights
        logger.info(
            "total datasets class_shots: {}, class_weights: {}".format(
                cls_shots, cls_weights))

        return D.ConcatDataset(datasets)


def make_data_loader(cfg, is_train=True):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert images_per_batch % num_gpus == 0, \
            "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used." \
                .format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert images_per_batch % num_gpus == 0, \
            "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used." \
                .format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    # aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    
    #import_file: import a module from the file path.
    path_catalog = import_file("smoke.config.paths_catalog", cfg.PATHS_CATALOG, True)
    DatasetCatalog = path_catalog.DatasetCatalog
    #transforms:  input:img & target    output:BGR,255 img & origin target
    transforms = build_transforms(cfg, is_train)
    dataset = build_dataset(cfg, transforms, DatasetCatalog, is_train)
    sampler = build_dataloder_sampler(cfg, dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler,
                                                          images_per_gpu,
                                                          drop_last=True)
    # return dict(images=images, targets=targets, img_ids=img_ids)
    collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader


def build_test_loader(cfg, is_train=False):
    path_catalog = import_file("smoke.config.paths_catalog", cfg.PATHS_CATALOG,
                               True)
    DatasetCatalog = path_catalog.DatasetCatalog

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        """
        Produce indices for inference.
        Inference needs to run on the __exact__ set of samples,
        therefore when the total number of samples is not divisible by the number of workers,
        this sampler produces different number of samples on different workers.
        """
        sampler = InferenceSampler(len(dataset))
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, cfg.TEST.IMS_PER_BATCH, drop_last=False
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)

    return data_loaders


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
