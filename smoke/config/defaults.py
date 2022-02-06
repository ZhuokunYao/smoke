import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.MODEL = CN()
_C.MODEL.SMOKE_ON = True
_C.MODEL.DEVICE = "cuda"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------

_C.INPUT = CN()
# Size of the smallest side of the image during training
"""
480
"""
_C.INPUT.HEIGHT_TRAIN = 384
# Maximum size of the side of the image during training
"""
640
"""
_C.INPUT.WIDTH_TRAIN = 1280
# Size of the smallest side of the image during testing
"""
480
"""
_C.INPUT.HEIGHT_TEST = 384
# Maximum size of the side of the image during testing
"""
640
"""
_C.INPUT.WIDTH_TEST = 1280
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.HEATMAP_METHORD = "round"           ### "round" or "oval"
"""
0.5
"""
_C.INPUT.RADIUS_IOU = 0.7
_C.INPUT.TEST_AFFINE_TRANSFORM = False
_C.INPUT.DEBUG_VISUAL = False

"""
new added
"""
_C.INPUT.SHIFT_SCALE_PROB_TRAIN = 0.5
_C.INPUT.SHIFT_SCALE_TRAIN = (0.2, 0.2)
_C.INPUT.FLIP_PROB_TRAIN = 0.5
_C.INPUT.MASK_REGRESS = False


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
"""
["jdx_fusion_front", "jdx2020_tracker", "jdx2021_tracker", "jdx2021_fusion"]
"""
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
"""
["jdx_simu_front", "jdx_test_front"]
"""
_C.DATASETS.TEST = ()
# train split tor dataset
_C.DATASETS.TRAIN_SPLIT = ""
# test split for dataset
_C.DATASETS.TEST_SPLIT = ""

"""
["Car", "Cyclist", "Pedestrian", "Truck", "Tricycle", "Bus"]
"""
_C.DATASETS.DETECT_CLASSES = ("Car",)
_C.DATASETS.MAX_OBJECTS = 100
_C.DATASETS.JDX_GAMMA_CORRECTION = False
# Setup params to filter special annotations
_C.DATASETS.DEPTH_RANGE = [0, 30]
# truncated [0, 1, 2, 3, 4, 5]
_C.DATASETS.TRUNCATION = [-1, 0, 1, 2, 3, 4, 5]
# occluded [0, 1, 2, 3, 4, 5]
_C.DATASETS.OCCLUSION = [-1, 0, 1, 2, 3, 4, 5]

_C.AUGMENTATION = CN()
_C.AUGMENTATION.TURN_ON = True
# Random HSV probability
_C.AUGMENTATION.HSV_PROB = 0.5
_C.AUGMENTATION.HSV_SCALE = (0.15, 0.7, 0.4)
# Flip probability
_C.AUGMENTATION.FLIP_PROB = 0.5
# Shift and scale probability
_C.AUGMENTATION.AFFINE_TRANSFORM_PROB = 0.3
_C.AUGMENTATION.AFFINE_TRANSFORM_SHIFT_SCALE = (0.2, 0.4)
# Mixup
_C.AUGMENTATION.MIXUP_PROB = 0.5
_C.AUGMENTATION.MIXUP_DATASET = ''
_C.AUGMENTATION.AUG_CLASS = ["Cyclist", "Truck", "Tricycle", "Bus"]
# CutMixup
_C.AUGMENTATION.CUTMIXUP_PROB = 0.5
# Each class should be cutted and saved separately
_C.AUGMENTATION.CUTMIXUP_DATASET = ''
_C.AUGMENTATION.CUTMIXUP_RANGE = (2, 8)

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
"""
0 why??
"""
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = False

_C.DATALOADER.SAMPLER = CN()
_C.DATALOADER.SAMPLER.TYPE = "TrainingSampler"
# The repeat factor just used for RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER.REPEAT_FACTOR = 0.3

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #

_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
"""
"RESNET-18"
"""
_C.MODEL.BACKBONE.CONV_BODY = "DLA-34-DCN"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 0
# Normalization for backbone
_C.MODEL.BACKBONE.USE_NORMALIZATION = "BN"
_C.MODEL.BACKBONE.DOWN_RATIO = 4
"""
train_smoke.py
if len(cfg.MODEL.BACKBONE.CHANNELS) > 0:
        cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS = cfg.MODEL.BACKBONE.CHANNELS[-1]
"""
_C.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS = 64
"""
train_smoke.py
after checkpointer = Checkpointer(...)
cfg.MODEL.BACKBONE.CHANNELS = checkpointer.load_param('backbone_channels')
"""
_C.MODEL.BACKBONE.CHANNELS = []

# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5

# ---------------------------------------------------------------------------- #
# Heatmap Head options
# ---------------------------------------------------------------------------- #

# --------------------------SMOKE Head--------------------------------
_C.MODEL.SMOKE_HEAD = CN()
_C.MODEL.SMOKE_HEAD.PREDICTOR = "SMOKEPredictor"
_C.MODEL.SMOKE_HEAD.LOSS_TYPE = ("GeneralizedFocalLoss", "DisL1")
_C.MODEL.SMOKE_HEAD.LOSS_ALPHA = 2
_C.MODEL.SMOKE_HEAD.LOSS_BETA = 4
_C.MODEL.SMOKE_HEAD.GENERALIZED_FL_BELTA = 2
# Channels for regression
# Specific channel for (depth_offset, keypoint_offset, dimension_offset, orientation)
_C.MODEL.SMOKE_HEAD.REGRESSION_HEADS = 8
_C.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL = (1, 2, 3, 2)
_C.MODEL.SMOKE_HEAD.REGRESSION_MULTI_HEADS = False
_C.MODEL.SMOKE_HEAD.USE_NORMALIZATION = "GN"
_C.MODEL.SMOKE_HEAD.NUM_CHANNEL = 256
# Loss weight for hm and reg loss
_C.MODEL.SMOKE_HEAD.LOSS_WEIGHT = (1., 1.)
"""
0.5
"""
_C.MODEL.SMOKE_HEAD.CLS_REWEIGHT_SHOT_GAMMA = 0.0
"""
0.5
"""
_C.MODEL.SMOKE_HEAD.CLS_REWEIGHT_SIZE_GAMMA = 0.0
# Reference car size in (length, height, width)
# for (car, cyclist, pedestrian, truck, tricycle, bus, cyclist_stopped)
"""
no cyclist_stopped
((4.392, 1.658, 1.910),
 (1.773, 1.525, 0.740),
 (0.505, 1.644, 0.582),
 (7.085, 2.652, 2.523),
 (2.790, 1.651, 1.201),
 (8.208, 2.869, 2.645))
"""
_C.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE = ((4.392, 1.658, 1.910),
                                           (1.773, 1.525, 0.740),
                                           (0.505, 1.644, 0.582),
                                           (7.085, 2.652, 2.523),
                                           (2.790, 1.651, 1.201),
                                           (8.208, 2.869, 2.645),
                                           (1.765, 1.185, 0.668))
# Reference depth
_C.MODEL.SMOKE_HEAD.DEPTH_REFERENCE = (28.01, 16.32)
# Reference normal focal is kitti, kitti:720, waymo:770, jdx:440, if < 0.0, not use focal reference
_C.MODEL.SMOKE_HEAD.NORMALIZED_FOCAL_REFERENCE = 720.0
_C.MODEL.SMOKE_HEAD.USE_NMS = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "Adam"
# ADAM
_C.SOLVER.BETAS = (0.9, 0.99)
# RMSProp
_C.SOLVER.ALPHA = 0.9

_C.SOLVER.SCHEDULER = 'MultiStepLR'
"""
200000
"""
_C.SOLVER.MAX_ITERATION = 14500
"""
[40000, 80000, 120000, 160000]
"""
_C.SOLVER.STEPS = (5850, 9350)
# CyclicLR
_C.SOLVER.CYCLICLR = CN()
"""
8e-4
"""
_C.SOLVER.CYCLICLR.BASE_LR = 1e-6
_C.SOLVER.CYCLICLR.MAX_LR = 1e-4
_C.SOLVER.CYCLICLR.STEPS_SIZE_UP = 1000
_C.SOLVER.CYCLICLR.STEPS_SIZE_DOWN = 1000
_C.SOLVER.CYCLICLR.MODE = 'triangular'
_C.SOLVER.CYCLICLR.SCALE_MODE = 'cycle'
#
_C.SOLVER.COSINE_TMAX = 5
_C.SOLVER.COSINE_ETAMIN = 0.00001
"""
0.5
"""
_C.SOLVER.DESCENT_RATE = 0.1
# WarmupMultiStepLR
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.BASE_LR = 0.00025
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.LOAD_OPTIMIZER_SCHEDULER = True

_C.SOLVER.CHECKPOINT_PERIOD = 20
_C.SOLVER.EVALUATE_STEP_PERIOD = 2000

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
"""
8
MUST BE DIVISIBLE BY THE NUMBER OF GPUS!!
"""
_C.SOLVER.IMS_PER_BATCH = 32
_C.SOLVER.MASTER_BATCH = -1

_C.SOLVER.PRETRAIN_MODEL = ''
_C.SOLVER.RESUME = False
# Specify the excluded layers when loading pretrained model
_C.SOLVER.EXCLUDE_LAYERS = []
# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.SINGLE_GPU_TEST = True
_C.TEST.IMS_PER_BATCH = 1
_C.TEST.PRED_2D = True

# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 50
_C.TEST.DETECTIONS_THRESHOLD = 0.25
"""
[[0,30],[30,60],[0,15],[15,30],[0,60]]
"""
_C.TEST.DEPTH_RANGES = [[0, 30], [30, 60]]
# use for evaluate map
_C.TEST.IOU_THRESHOLDS = [0.7, 0.25, 0.25, 0.7, 0.25, 0.7]

# ---------------------------------------------------------------------------- #
# Sparsity options
# ---------------------------------------------------------------------------- #
_C.PRUNNING = CN()
_C.PRUNNING.BN_SPARSITY = False
_C.PRUNNING.BN_RATIO = 0.001
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
"""
"./checkpoint/jdx_resnet18_640x480_crop_resize"
"""
_C.OUTPUT_DIR = "./tools/logs"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed does not
# guarantee fully deterministic behavior.
_C.SEED = 2020
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = True
_C.CONVERT_ONNX = False
_C.ENABLE_TENSORRT = False

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

