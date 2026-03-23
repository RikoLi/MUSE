from utils.tools import set_seed
from utils.logger import setup_logger
import os
import tqdm
import torch
import json
import argparse
import numpy as np
from configs import cfg
from utils.tools import create_top_k_similar_pairwise_dataset_enhanced, create_random_dataset
from utils.metrics import euclidean_distance
from models.reid_model import CLIP_ImageEncoder
from engine.datasets.dataloader import make_dataloader, make_val_dataloader

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
    if args.mode == 'train':
        dataloader, _, _, _, num_classes, num_cams, num_views = make_dataloader(cfg)
    elif args.mode == 'test':
        dataloader, num_queries = make_val_dataloader(cfg) # evaluate on given dataset only in single-source DG

    # initialize model
    model = CLIP_ImageEncoder(cfg, 1, 1, 1)
    model.load_param_inference(cfg.TEST.WEIGHT)

    # extract features
    features = []
    labels = []
    paths = []
    pids = []
    camids = []
    model.to("cuda")
    model.eval()
    for n_iter, (path, img, vid, camid, _) in enumerate(tqdm.tqdm(dataloader, desc=f"Extraction mode: {args.mode}")):
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                img = img.to("cuda")
                feat_bn = model.infer_image(img, after_bn=True)
        features.append(feat_bn.float().cpu())
        labels.append(vid)
        paths.extend(path)
        pids.append(vid)
        camids.append(camid)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    paths = np.array(paths)
    pids = torch.cat(pids, dim=0)
    camids = torch.cat(camids, dim=0)
    
    features = torch.nn.functional.normalize(features, dim=1, p=2)
    
    if args.mode == 'train' and args.pairwise_dataset_output_path != '':
        if args.random_sample:
            logger.info("Random select query-candidate pairs.")
            pairwise_trainset, pairwise_valset, train_ids, val_ids = create_random_dataset(paths, labels, args.k, args.val_id_ratio)
        else:
            logger.info("Using query-candidate hard mining.")
            pairwise_trainset, pairwise_valset, train_ids, val_ids = create_top_k_similar_pairwise_dataset_enhanced(features, paths, labels, args.k, val_id_ratio=args.val_id_ratio)
        suffix = args.pairwise_dataset_output_path.split('.')[-1]
        save_path = f"{os.path.join(*args.pairwise_dataset_output_path.split('.')[:-1])}_train.{suffix}"
        with open(save_path, 'w') as f:
            json.dump(pairwise_trainset, f, indent=4)
        save_path = f"{os.path.join(*args.pairwise_dataset_output_path.split('.')[:-1])}_val.{suffix}"
        with open(save_path, 'w') as f:
            json.dump(pairwise_valset, f, indent=4)
        logger.info(f"Pair dataset saved to {args.pairwise_dataset_output_path}")
        logger.info(f'Train #IDs: {len(train_ids)}')
        logger.info(f'Val #IDs: {len(val_ids)}')
    elif args.mode == 'test' and args.pairwise_dataset_output_path != '':
        raise NotImplementedError("Test pairwise dataset generation is disabled now.")
    
    if args.distmat_output_path != '':
        distmat = euclidean_distance(features[:num_queries], features[num_queries:], return_tensor=False)
        np.savez(args.distmat_output_path, distmat=distmat, paths=paths, pids=pids, camids=camids)
        logger.info(f"Distmat saved to {args.distmat_output_path}")

    if args.feature_output_path != '':
        np.savez(args.feature_output_path, features=features, paths=paths, pids=pids, camids=camids)
        logger.info(f"Features saved to {args.feature_output_path}")

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", help="path to config file", type=str)
    parser.add_argument("--pairwise-dataset-output-path", default="", help="path to save output pairwise dataset", type=str)
    parser.add_argument("--distmat-output-path", default="", help="path to save output distmat file", type=str)
    parser.add_argument('--feature-output-path', type=str, default='', help='path to save output feature file')
    parser.add_argument("--k", default=5, help="sample numbers k", type=int)
    parser.add_argument("--val-id-ratio", default=0.5, help="ratio of validation IDs", type=float)
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'test'], help="train or test mode")
    parser.add_argument("--random-sample", action='store_true', help="randomly select query-candidate pairs")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    main(args)