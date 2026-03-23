from utils.tools import set_seed
from utils.logger import setup_logger
import os
import torch
import argparse
from configs import cfg
from models.reid_model import CLIP_ImageEncoder
from utils.metrics import R1_mAP_eval
from engine.datasets.dataloader import make_dataloader, make_val_dataloader
from engine.solvers.optimizers import make_optimizer_baseline
from engine.solvers.schedulers import WarmupMultiStepLR
from engine.trainers.baseline_trainer import BaselineTrainer

def main(args):
    # load config
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # set random seed and other logger
    set_seed(cfg.SOLVER.SEED)
    logger = setup_logger("logger", cfg.OUTPUT_DIR, if_train=True)
    logger.info("Running with config:\n{}".format(cfg))
    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # create dataloaders
    train_loader_normal, train_loader, val_loader, num_queries, num_classes, num_cams, num_views = make_dataloader(cfg)
    if len(cfg.DATASETS.NAMES) == 1 and cfg.DATASETS.EVAL_DATASET != "":
        logger.info(f"Use {cfg.DATASETS.EVAL_DATASET} for evaluation.")
        val_loader, num_queries = make_val_dataloader(cfg) # evaluate on given dataset only in single-source DG

    # initialize model
    model = CLIP_ImageEncoder(cfg, num_classes, num_cams, num_views)

    # initialize optimizer
    optimizer = make_optimizer_baseline(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer,
                                  milestones=cfg.SOLVER.STEPS,
                                  gamma=cfg.SOLVER.GAMMA,
                                  warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                                  warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                                  warmup_method=cfg.SOLVER.WARMUP_METHOD)

    # trainer
    trainer = BaselineTrainer(cfg,
                          logger=logger,
                          device=cfg.MODEL.DEVICE,
                          model=model,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          evaluator=R1_mAP_eval(num_queries,
                                                max_rank=50,
                                                feat_norm=cfg.TEST.FEAT_NORM,
                                                reranking=cfg.TEST.RE_RANKING))
    # test mode
    if args.test:
        trainer.inference(val_loader, cfg.TEST.WEIGHT)
    # train mode
    else:
        trainer.train(train_loader, train_loader_normal, val_loader)

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", help="path to config file", type=str)
    parser.add_argument("--test", action="store_true", help="perform evaluation only")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    main(args)