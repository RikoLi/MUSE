import argparse
import numpy as np
from utils.metrics import eval_func

def main(args):
    data = np.load(args.distmat_path)
    distmat = data['distmat']
    paths = data['paths']
    pids = data['pids']
    camids = data['camids']
    num_queries = len([p for p in paths if 'query' in p])
    
    print(f"Dist matrix shape: {distmat.shape} from {args.distmat_path}")
    print(f'Evaluating...')
    
    cmc, mAP = eval_func(distmat, pids[:num_queries], pids[num_queries:], camids[:num_queries], camids[num_queries:])
    
    print(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        print(f"CMC curve, Rank-{r:<3}:{cmc[r-1]:.1%}")

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--distmat-path', type=str, default='', help='Path to the distmat file')
    args = parser.parse_args()
    
    main(args)