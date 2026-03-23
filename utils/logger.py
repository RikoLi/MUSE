import logging
import os
import sys
import time
import os.path as osp

def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.log"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.log"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def timestamp_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        log_file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        fh = logging.FileHandler(os.path.join(save_dir, f"{log_file_name}.log"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger