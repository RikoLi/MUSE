import torch
import logging
from utils.tools import format_number

logger = logging.getLogger("logger")

def make_optimizer_baseline(cfg, model):
    params = []
    param_keys = []
    param_cnt = 0
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    
    for key, value in model.named_parameters():
        if value.requires_grad:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            param_cnt += value.numel()
            param_keys.append(key)
    
    logger.info(f"Trainable params: {param_keys}")
    logger.info(f'Total {format_number(param_cnt)} trainable params in {format_number(len(param_keys))} layers.')
        
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        
    return optimizer

def make_optimizer(cfg, model, accelerator):
    params = []
    param_keys = []
    param_cnt = 0
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    
    for key, value in model.named_parameters():
        if value.requires_grad:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            param_cnt += value.numel()
            param_keys.append(key)
    
    if accelerator.is_main_process:
        logger.info(f"Trainable params: {param_keys}")
        logger.info(f'Total {format_number(param_cnt)} trainable params in {format_number(len(param_keys))} layers.')
        
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        
    return optimizer
