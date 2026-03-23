import tqdm
import numpy as np
import torch
from sklearn import metrics
from scipy.sparse import csr_matrix

def log_div(x, y):
    x.clip_(min=1e-8)
    y.clip_(min=1e-8)
    x.log_()
    y.log_()
    x.sub_(y)
    x.exp_()
    return x

def enhance_original_dist(num_query, original_dist, extra_dist, gallery_indices, scale, fuse_alpha=0.5):
    extra_dist = scale * extra_dist.type_as(original_dist)
    fused_dist = (1 - fuse_alpha) * original_dist[:num_query, num_query:].gather(1, gallery_indices) + fuse_alpha * extra_dist
    original_dist.scatter_(1, gallery_indices+num_query, fused_dist)
    original_dist[num_query:, :num_query] = original_dist[:num_query, num_query:].t() # Symmetry
    return original_dist

def re_ranking(feat, query_num, k1, k2, lambda_value, extra_dist=None, swapped_dist=None, gallery_indices=None, fuse_alpha=0.0):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    all_num = feat.size(0)
    
    # Compute original distance matrix
    feat = feat.to(torch.float16).numpy()
    original_dist = metrics.pairwise.pairwise_distances(feat, feat, metric='euclidean', n_jobs=-1).astype(np.float16, copy=False) # range [0, 2]
    original_dist = torch.from_numpy(original_dist)
    del feat
    if extra_dist is not None:
        extra_dist = extra_dist.to(torch.float16)
        if swapped_dist is not None:
            swapped_dist = swapped_dist.to(torch.float16)
            extra_dist = 0.5 * (extra_dist + swapped_dist) # swapped query-candidate enhancement
        original_dist = enhance_original_dist(query_num, original_dist, extra_dist, gallery_indices, scale=2, fuse_alpha=fuse_alpha) # scale is  (maximal value of original distance matrix)
        print(f'=> Enhanced by extra distance matrix with fusion alpha {fuse_alpha}')
    gallery_num = original_dist.shape[0]
    
    # Normalize original distance matrix
    original_dist = log_div(original_dist, original_dist.max(dim=0)[0]).transpose(0, 1).to(torch.float16)
    print('=> Original distmat is computed')
    V = torch.empty_like(original_dist).zero_()
    print('=> V is initialized')
    initial_rank = torch.argsort(original_dist)
    
    print('=> Starting re-ranking')
    for i in tqdm.tqdm(range(all_num), desc='Re-ranking'):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1] # Top k neighbors of current point
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1] # Top k neighbors of current point's top k neighbors
        fi = torch.where(backward_k_neigh_index == i)[0] # Index positions of mutual nearest neighbors
        k_reciprocal_index = forward_k_neigh_index[fi] # Indices of mutual nearest neighbors
        
        # Expand k-reciprocal neighbors: R -> R*
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j] # Each mutual nearest neighbor sample

            # Expand R by k/2
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(round(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(round(k1 / 2)) + 1]
            fi_candidate = torch.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            intersection = torch.from_numpy(np.intersect1d(candidate_k_reciprocal_index.numpy(), k_reciprocal_index.numpy()))
            if len(intersection) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = torch.from_numpy(np.append(k_reciprocal_expansion_index.numpy(), candidate_k_reciprocal_index.numpy()))
        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)

        # Vectorization + normalization
        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)
    
    # Extract first query rows
    original_dist = original_dist[:query_num, ]

    # local query expansion
    if k2 != 1:
        V_qe = torch.zeros_like(V, dtype=torch.float16)
        for i in range(all_num):
            V_qe[i, :] = torch.mean(V[initial_rank[i, :k2], :], dim=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    
    # gallery_num here refers to all gallery+query
    for i in range(gallery_num):
        invIndex.append(torch.where(V[:, i] != 0)[0]) # Indices of samples with non-zero similarity for each sample

    # Compute Jaccard distance
    jaccard_dist = torch.zeros_like(original_dist, dtype=torch.float16)
    for i in range(query_num):
        temp_min = torch.zeros((1, gallery_num), dtype=torch.float16)
        indNonZero = torch.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + torch.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_ranking_ca_jaccard(feat, query_num, cids, k1, k2, k1_intra, k1_inter, k2_intra, k2_inter, ckrnns=False, clqe=False, lambda_value=0.3,
                          extra_dist=None, gallery_indices=None, fuse_alpha=0.0):
    if ckrnns and clqe:
        mode = f"[CAJaccard (CKRNNS + CLQE)]"
    elif ckrnns and not clqe:
        mode = f"[CAJaccard (CKRNNS + LQE)]"
    elif not ckrnns and clqe:
        mode = f"[CAJaccard (KRNNS + CLQE)]"
    else:
        mode = f"[Jaccard (KRNNS + LQE)]"
    print(mode)

    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    all_num = feat.size(0)
    
    # Compute original distance matrix
    feat = feat.to(torch.float16).numpy()
    original_dist = metrics.pairwise.pairwise_distances(feat, feat, metric='euclidean', n_jobs=-1).astype(np.float16, copy=False) # range [0, 4]
    original_dist = torch.from_numpy(original_dist)
    del feat
    if extra_dist is not None:
        extra_dist = extra_dist.to(torch.float16)
        original_dist = enhance_original_dist(query_num, original_dist, extra_dist, gallery_indices, scale=2, fuse_alpha=fuse_alpha) # scale is 2 (maximal value of original distance matrix)
        print(f'=> Enhanced by extra distance matrix with fusion alpha {fuse_alpha}')
    gallery_num = original_dist.shape[0]
    
    # Normalize original distance matrix
    original_dist = log_div(original_dist, original_dist.max(dim=0)[0]).transpose(0, 1).to(torch.float16)
    print('=> Original distmat is computed')
    V = torch.empty_like(original_dist).zero_()
    print('=> V is initialized')
    initial_rank = torch.argsort(original_dist)

    cam_mask = (cids.reshape(-1, 1) == cids.reshape(1, -1))

    inter_rank = np.argpartition(original_dist.numpy() + 999.0 * cam_mask, range(k1_inter + 2))
    nn_inter = [k_reciprocal_neigh(inter_rank, i, k1_inter) for i in range(all_num)]
    print('=> nn_inter is computed')
    del inter_rank
    intra_rank = np.argpartition(original_dist.numpy() + 999.0 * (~cam_mask), range(k1_intra + 2))
    nn_intra = [k_reciprocal_neigh(intra_rank, i, k1_intra) for i in range(all_num)]
    print('=> nn_intra is computed')
    del intra_rank

    del cam_mask

    # inter_rank = torch.from_numpy(inter_rank).to(torch.float16)
    # intra_rank = torch.from_numpy(intra_rank).to(torch.float16)
    # nn_inter = torch.from_numpy(nn_inter).to(torch.float16)
    # nn_intra = torch.from_numpy(nn_intra).to(torch.float16)

    ###################################
    #           KRNNs/CKRNNs          #
    ###################################
    if ckrnns:
        print(f"[CKRNNs] PARAMS: k1_intra: {k1_intra}, k1_inter: {k1_inter}")
    else:
        print(f"[KRNNs] PARAMS: k1: {k1}")

    for i in tqdm.tqdm(range(all_num), desc='Re-ranking'):
        if ckrnns:
            k_reciprocal_index = torch.from_numpy(np.append(nn_intra[i], nn_inter[i]))
            k_reciprocal_expansion_index = k_reciprocal_index
        else:
            forward_k_neigh_index = initial_rank[i, :k1 + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
            fi = torch.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate, :int(round(k1 / 2)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                   :int(round(k1 / 2)) + 1]
                fi_candidate = torch.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(torch.from_numpy(np.intersect1d(candidate_k_reciprocal_index.numpy(), k_reciprocal_index.numpy()))) > 2. / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = torch.from_numpy(np.append(k_reciprocal_expansion_index.numpy(), candidate_k_reciprocal_index.numpy()))

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)
        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)
    original_dist = original_dist[:query_num, ]
    ################################
    #            LQE/CLQE          #
    ################################
    V_qe = torch.zeros_like(V, dtype=torch.float16)
    if clqe:
        print(f"[CLQE] PARAMS: k2_intra: {k2_intra}, k2_inter: {k2_inter}")
    else:
        print(f"[LQE] PARAMS: k2: {k2}")

    for i in range(all_num):
        if clqe:
            k2nn = torch.from_numpy(np.append(intra_rank[i, :k2_intra].numpy(), inter_rank[i, :k2_inter].numpy()))
        else:
            k2nn = initial_rank[i, :k2]
        V_qe[i, :] = torch.mean(V[k2nn, :], dim=0)
    V = V_qe
    del V_qe
    # del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(torch.where(V[:, i] != 0)[0])

    jaccard_dist = torch.zeros_like(original_dist, dtype=torch.float16)

    for i in range(query_num):
        temp_min = torch.zeros((1, gallery_num), dtype=torch.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + torch.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def rankdist(initial_rank, k):
    pos_L1 = torch.from_numpy(initial_rank).argsort().numpy().astype(np.int32)
    fac_1 = csr_matrix(np.maximum(0, k - pos_L1))
    rankdist = fac_1 @ fac_1.T
    return -rankdist.toarray()

def merge_dist(ecn_dist, orig_dist, nQuery):
    orig_dist = orig_dist[nQuery:, :nQuery]
    ecn_dist = np.where(ecn_dist != 0, ecn_dist, orig_dist)
    return ecn_dist

def re_ranking_ecn(feat, query_num, k=25, t=3, q=8, method='rankdist', extra_dist=None, gallery_indices=None, fuse_alpha=0.0):
    print(f'ECN params k: {k}, t: {t}, q: {q}')
    feat = feat.to(torch.float16).numpy()
    num_galleries = feat.shape[0] - query_num
    orig_dist = metrics.pairwise.pairwise_distances(feat, feat, metric='euclidean', n_jobs=-1).astype(np.float16, copy=False) # range [0, 2]
    orig_dist = torch.from_numpy(orig_dist)
    
    if extra_dist is not None:
        extra_dist = extra_dist.to(torch.float16)
        orig_dist = enhance_original_dist(query_num, orig_dist, extra_dist, gallery_indices, scale=2, fuse_alpha=fuse_alpha) # scale is 2 (maximal value of original distance matrix)
        print(f'=> Enhanced by extra distance matrix with fusion alpha {fuse_alpha}')
    
    initial_rank = orig_dist.argsort().numpy()

    if method == 'rankdist':
        r_dist = rankdist(initial_rank, k)
        print('rankdist computed...commencing ECN')
    else:
        r_dist = orig_dist

    top_t_nb = initial_rank[:, 1:t + 1]
    t_ind = top_t_nb[query_num:, :].T
    next_2_tnbr = np.transpose(initial_rank[t_ind, 1:q + 1], [0, 2, 1])
    next_2_tnbr = np.reshape(next_2_tnbr, (t * q, num_galleries))
    t_ind = np.concatenate((t_ind, next_2_tnbr), axis=0)

    q_ind = top_t_nb[:query_num, :].T
    next_2_qnbr = np.transpose(initial_rank[q_ind, 1:q + 1], [0, 2, 1])
    next_2_qnbr = np.reshape(next_2_qnbr, (t * q, query_num))

    q_ind = np.concatenate((q_ind, next_2_qnbr), axis=0)

    t_nbr_dist = r_dist[t_ind, :query_num]

    q_nbr_dist = r_dist[q_ind, query_num:]
    q_nbr_dist = np.transpose(q_nbr_dist, [0, 2, 1])

    ecn_dist = np.mean(np.concatenate((q_nbr_dist, t_nbr_dist), axis=0), axis=0)

    # Use orig_dist values where rank-list based similarities are zero -- fixes behaviour in large scale open-ended retrievals
    ecn_dist = merge_dist(ecn_dist, orig_dist, query_num)
    print('ECN dist compute done...')
    return ecn_dist.T