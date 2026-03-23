from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# Name of backbone
_C.MODEL.NAME = 'ViT-B-16'
# Loss weights
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.PCL_LOSS_WEIGHT = 1.0
_C.MODEL.MATCH_LOSS_WEIGHT = 1.0
# Transformer setting
_C.MODEL.STRIDE_SIZE = [16, 16]
# freeze patch projection
_C.MODEL.FREEZE_PATCH_PROJ = True
# camera embedding
_C.MODEL.ENABLE_CAM_EMB = False
# checkpoint
_C.MODEL.CHECKPOINT = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501',)
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
# Evaluate on another dataset
_C.DATASETS.EVAL_DATASET = ''
# Use new MSDG protocol
_C.DATASETS.USE_NEW_MSDG_PROTOCOL = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
_C.SOLVER = CN()
_C.SOLVER.SEED = 1234
_C.SOLVER.IMS_PER_BATCH = 64
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 1e-4

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_LR_INIT = 0.01
_C.SOLVER.LR_MIN = 0.000016
_C.SOLVER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10

# PCL memory
_C.PCL = CN()
_C.PCL.MEMORY_MOMENTUM = 0.2
_C.PCL.HARD_MEMORY_UPDATE = True
_C.PCL.MEMORY_TEMP = 0.01

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 256
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
_C.TEST.REID_ENCODER_WEIGHT = ""
_C.TEST.FEAT_NORM = True
_C.TEST.AFTER_BN = True
_C.TEST.MLLM_RERANKING_LAMBDA = 0.7

# ---------------------------------------------------------------------------- #
# MLLM
# ---------------------------------------------------------------------------- #
_C.MLLM = CN()
_C.MLLM.BASE_MODEL = "qwen2.5vl-3b-instruct"
_C.MLLM.PRETRAINED_ROOT = ""
_C.MLLM.POS_K = 1
_C.MLLM.NEG_K = 1
_C.MLLM.TOPK = 10
_C.MLLM.FINETUNE_MODE = "freeze_llm"
_C.MLLM.LORA_RANK = 8
_C.MLLM.LORA_ALPHA = 32
_C.MLLM.LORA_DROPOUT = 0.05
_C.MLLM.ENABLE_GRADIENT_CHECKPOINT = False
_C.MLLM.USE_PARTFORMER = False
_C.MLLM.PARTFORMER_N_PARTS = 4
_C.MLLM.PARTFORMER_DEPTH = 2
_C.MLLM.PARTFORMER_HEADS = 8
_C.MLLM.PARTFORMER_DROPOUT = 0.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""