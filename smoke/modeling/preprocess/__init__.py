from .preprocess import Normalization

def build_preprocess(cfg):
    return Normalization(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)