import time
from datetime import timedelta
import argparse
import torch
import numpy as np
from sklearn import metrics
from utils.metrics import eval_func
from utils.reranking_fast import re_ranking, re_ranking_ecn, re_ranking_ca_jaccard

def main(args):
    data = np.load(args.test_feature_path)
    features = data['features']
    pids = data['pids']
    camids = data['camids']
    num_queries = len([path for path in data['paths'] if 'query' in path])
    print(f'Number of queries: {num_queries}')

    features = torch.from_numpy(features)
    features = torch.nn.functional.normalize(features, dim=1, p=2)  # along channel

    # query
    q_pids = np.asarray(pids[:num_queries])
    q_camids = np.asarray(camids[:num_queries])

    # gallery
    g_pids = np.asarray(pids[num_queries:])
    g_camids = np.asarray(camids[num_queries:])

    # load extra distmat if available
    if args.extra_distmat != '':
        data = np.load(args.extra_distmat)
        extra_dist = data['probs']
        gallery_indices = data['gallery_indices']
        extra_dist = torch.from_numpy(extra_dist)
        gallery_indices = torch.from_numpy(gallery_indices)
        print('=> Load extra distance matrix')
    else:
        extra_dist = None
        gallery_indices = None
        
    print(f'=> Enter reranking: {args.reranking_mode}')
    start_time = time.monotonic()
    if args.reranking_mode == 'simple':
        features = features.to(torch.float16)
        extra_dist = extra_dist.to(torch.float16)
        distmat = metrics.pairwise_distances(features.numpy(), features.numpy(), metric='euclidean', n_jobs=-1).astype(np.float16, copy=False)
        distmat = torch.from_numpy(distmat)
        distmat = distmat[:num_queries, num_queries:]
        fused_dist = (1 - args.fuse_alpha) * distmat.gather(1, gallery_indices) + args.fuse_alpha * extra_dist * 2.0
        distmat.scatter_(1, gallery_indices, fused_dist)
    elif args.reranking_mode == 'krnn':
        distmat = re_ranking(features, num_queries, k1=10, k2=5, lambda_value=args.lambda_value,
                             extra_dist=extra_dist,
                             gallery_indices=gallery_indices,
                             fuse_alpha=args.fuse_alpha)
    elif args.reranking_mode == 'ecn':
        distmat = re_ranking_ecn(features, num_queries, k=25, extra_dist=extra_dist, gallery_indices=gallery_indices, fuse_alpha=args.fuse_alpha)
    elif args.reranking_mode == 'caj':
        distmat = re_ranking_ca_jaccard(features, num_queries, cids=camids, k1=10, k2=5,
                                        k1_intra=5, k1_inter=20,
                                        k2_intra=2, k2_inter=4,
                                        ckrnns=True,
                                        clqe=False, # "True" to use CLQE in CAJ for better performance
                                        lambda_value=args.lambda_value,
                                        extra_dist=extra_dist,
                                        gallery_indices=gallery_indices,
                                        fuse_alpha=args.fuse_alpha)
    else:
        raise ValueError(f"Invalid reranking mode: {args.reranking_mode}")

    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    duration = timedelta(seconds=time.monotonic() - start_time)

    print(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        print(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
    print(f"Reranking time: {duration}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-feature-path', type=str, help='Path to the test feature file')
    parser.add_argument('--extra-distmat', type=str, default='', help='Path to the extra distance matrix file')
    parser.add_argument('--lambda-value', type=float, default=0.3, help='lambda value for reranking')
    parser.add_argument('--fuse-alpha', type=float, default=0.2, help='alpha value for extra dist fusion')
    parser.add_argument('--reranking-mode', type=str, choices=['simple', 'krnn', 'ecn', 'caj'], default='krnn', help='Reranking mode')
    args = parser.parse_args()
    main(args)