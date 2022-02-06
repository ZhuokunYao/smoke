import logging

from .augmentations import *

logger = logging.getLogger("augmentations")


def build_augmentations(cfg):
    augmentations = Compose(
        [
            #  0.5, (0.15, 0.7, 0.4)
            RandomHSV(cfg.AUGMENTATION.HSV_PROB, cfg.AUGMENTATION.HSV_SCALE[0], cfg.AUGMENTATION.HSV_SCALE[1], cfg.AUGMENTATION.HSV_SCALE[2]),
            #  0.5
            RandomHorizontallyFlip(cfg.AUGMENTATION.FLIP_PROB),
            RandomAffineTransformation(cfg.AUGMENTATION.AFFINE_TRANSFORM_PROB,  # 0.3
                                       cfg.AUGMENTATION.AFFINE_TRANSFORM_SHIFT_SCALE, #（0.2， 0.4）
                                       [cfg.INPUT.WIDTH_TRAIN,
                                        cfg.INPUT.HEIGHT_TRAIN],
                                       cfg.DATASETS.DETECT_CLASSES[1:]),
        ]
    )
    return augmentations
