import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
# Base config files
_C.BASE = ['']

_C.DIS = False
_C.WORLD_SIZE = 1
_C.SEED = 1234
_C.AMP = True
_C.EXPERIMENT_ID = ""
_C.SAVE_DIR = "save_pth"
_C.MODEL_PATH = ""

_C.WANDB = CN()
_C.WANDB.PROJECT = "CVSS_FSL"
_C.WANDB.TAG = ""
_C.WANDB.MODE = "offline"
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN_IMAGE_PATH = "/ai/data/data/vessel/DIAS/DSA/data/training/images"
_C.DATASET.TRAIN_LABEL_PATH = "/ai/data/data/vessel/DIAS/DSA/data/training/labels"
_C.DATASET.VAL_IMAGE_PATH = "/ai/data/data/vessel/DIAS/DSA/data/validation/images"
_C.DATASET.VAL_LABEL_PATH = "/ai/data/data/vessel/DIAS/DSA/data/validation/labels"
_C.DATASET.TEST_IMAGE_PATH = "/ai/data/data/vessel/DIAS/DSA/data/test/images"
_C.DATASET.TEST_LABEL_PATH = "/ai/data/data/vessel/DIAS/DSA/data/test/labels"
_C.DATASET.STRIDE = 32
_C.DATASET.PATCH_SIZE = (64, 64)
_C.DATASET.NUM_EACH_EPOCH = 20000
_C.DATASET.WITH_VAL = True

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 64
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = "FR_UNet"

_C.TRAIN = CN()
_C.TRAIN.DO_BACKPROP = False
_C.TRAIN.VAL_NUM_EPOCHS = 1
_C.TRAIN.SAVE_PERIOD = 1
_C.TRAIN.MNT_MODE = "max"
_C.TRAIN.MNT_METRIC = "AUC"
_C.TRAIN.EARLY_STOPPING = 100

_C.TRAIN.EPOCHS = 100
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'

# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if args.cfg is not None:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATALOADER.BATCH_SIZE = args.batch_size
    if args.model_type:
        config.MODEL.TYPE = args.model_type
    if args.tag:
        config.WANDB.TAG = config.MODEL.TYPE + "_" + args.tag
    else:
        config.WANDB.TAG = config.MODEL.TYPE
    if args.wandb_mode == "online":
        config.WANDB.MODE = args.wandb_mode
    if args.world_size:
        config.WORLD_SIZE = args.world_size
    if args.enable_distributed:
        config.DIS = True
    config.freeze()


def update_val_config(config, args):
    if args.cfg is not None:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.model_path:
        config.MODEL_PATH = args.model_path
    config.freeze()


def get_config(args=None):
    config = _C.clone()
    update_config(config, args)

    return config


def get_config_no_args():
    config = _C.clone()

    return config


def get_val_config(args=None):
    config = _C.clone()
    update_val_config(config, args)

    return config
