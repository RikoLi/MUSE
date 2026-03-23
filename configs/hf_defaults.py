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
# checkpoint
_C.MODEL.CHECKPOINT = ""
_C.MODEL.PRETRAINED_MODEL_DIR = ""
_C.MODEL.EXTRA_TRAINABLE_LAYERS = []
_C.MODEL.DDP_TIMEOUT_HOURS = 3 # default DDP timeout: 3 hours

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.IMAGE_SIZE = [280, 140]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAME = ""
_C.DATASETS.PROMPT_TEMPLATE = ""
_C.DATASETS.PAIRWISE_TRAINSET_PATH = ""
_C.DATASETS.PAIRWISE_TRAINSET_TRUNCATE_RATIO = 1.0
_C.DATASETS.PAIRWISE_VALSET_PATH = ""
_C.DATASETS.PAIRWISE_TESTSET_PATH = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.MAX_ITER = -1

# ---------------------------------------------------------------------------- #
# Solver
_C.SOLVER = CN()
_C.SOLVER.SEED = 1234
_C.SOLVER.BATCHSIZE = 2
_C.SOLVER.OPTIMIZER_NAME = "AdamW"
_C.SOLVER.MAX_EPOCHS = 4
_C.SOLVER.BASE_LR = 5e-5
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.LR_MIN = 1e-7
_C.SOLVER.WARMUP_LR_INIT = 1e-5
_C.SOLVER.WARMUP_T = 2
_C.SOLVER.EVAL_PERIOD = 2
_C.SOLVER.CHECKPOINT_PERIOD = 2
_C.SOLVER.TBX_LOG_PERIOD = 10

_C.LORA = CN()
_C.LORA.TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_C.LORA.RANK = 16
_C.LORA.ALPHA = 32
_C.LORA.DROPOUT = 0.05

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
_C.TEST.BATCHSIZE = 2

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""