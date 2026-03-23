import os
import sys
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import numpy as np
import random
import time
import logging
import tqdm
from datetime import timedelta

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_number(num):
    """
    Format a number by converting it to a string with units (K, M, B, etc.).
    
    Args:
        num (int or float): The number to format.
        
    Returns:
        str: The formatted string.
    """
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)
    
def save_running_config(cfg, output_dir):
    new_cfg = cfg.clone()
    new_cfg.defrost()
    new_cfg.OUTPUT_DIR = ''
    new_cfg.freeze()
    save_path = os.path.join(output_dir, 'running_config.yaml')
    with open(save_path, 'w') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print(new_cfg)
        sys.stdout = old_stdout
    print('Running config is saved.')

def set_partial_layer_trainable(model, trainable_layers):
    """
    Set partial layers of the model to trainable state.
    
    Args:
        model (nn.Module): The model to configure.
        trainable_layers (List[str]): List of layer names to set as trainable.
    """
    for name, param in model.named_parameters():
        for layer_name in trainable_layers:
            if layer_name in name:
                param.requires_grad_(True)
                break

def compute_cluster_centroids(features, labels, l2_norm=True):
    """
    Compute L2-normed cluster centroid for each class.
    """
    num_classes = len(labels.unique()) - 1 if -1 in labels else len(labels.unique())
    centers = torch.zeros((num_classes, features.shape[1]), dtype=torch.float32)
    for i in range(num_classes):
        idx = torch.where(labels == i)[0]
        temp = features[idx,:]
        if len(temp.shape) == 1:
            temp = temp.reshape(1, -1)
        centers[i,:] = temp.mean(0)
    return F.normalize(centers, dim=1) if l2_norm else centers

def create_top_k_similar_pairwise_dataset(reid_embeds, image_paths, labels, pos_k, neg_k, format):
    similarity_matrix = torch.mm(reid_embeds, reid_embeds.t())
    mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
    similarity_matrix.masked_fill_(mask, -float('inf'))
    
    results = []
    for i in range(similarity_matrix.size(0)):
        query_image_path = image_paths[i]
        query_label = labels[i]
        
        # Get indices of same-class samples
        same_class_mask = (labels == query_label)
        same_class_mask[i] = False  # Exclude self
        same_class_similarity = similarity_matrix[i][same_class_mask]
        same_class_indices = torch.where(same_class_mask)[0]
        
        # Get indices of different-class samples
        diff_class_mask = (labels != query_label)
        diff_class_similarity = similarity_matrix[i][diff_class_mask]
        diff_class_indices = torch.where(diff_class_mask)[0]
        
        # Get top k least similar samples from same class
        _, pos_k_indices = torch.topk(same_class_similarity, min(pos_k, len(same_class_similarity)), dim=0, largest=False)
        
        # Get top k most similar samples from different class
        _, neg_k_indices = torch.topk(diff_class_similarity, min(neg_k, len(diff_class_similarity)), dim=0, largest=True)
        
        # Process same-class samples
        for j in range(min(pos_k, len(pos_k_indices))):
            candidate_index = same_class_indices[pos_k_indices[j]].item()
            candidate_image_path = image_paths[candidate_index]

            if format == "llamafactory":
                result = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Query image: <image>\nCandidate image: <image>\nDo the people in above two images have the same identity? Answer yes or no according to the criteria."
                        },
                        {
                            "from": "gpt",
                            "value": "yes"
                        }
                    ],
                    "images": [
                        query_image_path,
                        candidate_image_path
                    ],
                    "system": "You are an expert in person recognition. You always focus on person attributes like appearance, age, gender, body parts and clothing. " + \
                            "You are robust across diverse environments and never disturbed by identity irrelevant factors like background, illumination and occlusion. " + \
                            "The criteria to distinguish two identities include: 1. appearance (age, gender, hair style, hair color, face, shape of body), " + \
                            "2. attirement (upper or lower body clothing type, color, material, accessory, body equipments), 3. carried object (handbag, backpack, umbrella, object in hand)."
                }
            elif format == "vanilla":
                result = {"query": query_image_path, "candidate": candidate_image_path, "match": "Yes"}
            results.append(result)
        
        # Process different-class samples
        for j in range(min(neg_k, len(neg_k_indices))):
            candidate_index = diff_class_indices[neg_k_indices[j]].item()
            candidate_image_path = image_paths[candidate_index]
            
            if format == "llamafactory":
                result = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Query image: <image>\nCandidate image: <image>\nDo the people in above two images have the same identity? Answer yes or no according to the criteria."
                        },
                        {
                            "from": "gpt",
                            "value": "no"
                        }
                    ],
                    "images": [
                        query_image_path,
                        candidate_image_path
                    ],
                    "system": "You are an expert in person recognition. You always focus on person attributes like appearance, age, gender, body parts and clothing. " + \
                            "You are robust across diverse environments and never disturbed by identity irrelevant factors like background, illumination and occlusion. " + \
                            "The criteria to distinguish two identities include: 1. appearance (age, gender, hair style, hair color, face, shape of body), " + \
                            "2. attirement (upper or lower body clothing type, color, material, accessory, body equipments), 3. carried object (handbag, backpack, umbrella, object in hand)."
                }
            elif format == "vanilla":
                result = {"query": query_image_path, "candidate": candidate_image_path, "match": "No"}
            results.append(result)
    
    return results

def create_random_dataset(image_paths, labels, k, val_id_ratio):
    # Get all unique IDs
    unique_ids = labels.unique()
    num_val_ids = int(len(unique_ids) * val_id_ratio)
    
    # Randomly select validation set IDs
    val_ids = set(random.sample(unique_ids.tolist(), num_val_ids))
    train_ids = set(unique_ids.tolist()) - val_ids
    
    # Store training and validation set results separately
    train_results = []
    val_results = []
    
    all_indices = torch.arange(len(image_paths))
    for i in range(len(image_paths)):
        query_image_path = image_paths[i]
        query_label = labels[i]
        
        # Randomly sample k samples different from query
        indices = torch.where(all_indices != i)[0]
        sample_indices = random.sample(indices.tolist(), k)
                
        for idx in sample_indices:
            result = {"query": query_image_path, "candidate": image_paths[idx], "match": "Yes" if query_label.item() == labels[idx] else "No"}
            if query_label.item() in val_ids:
                val_results.append(result)
            else:
                train_results.append(result)
                
    return train_results, val_results, train_ids, val_ids

def create_top_k_similar_pairwise_dataset_enhanced(reid_embeds, image_paths, labels, k, val_id_ratio):
    """
    1. Split training set and validation set by ID
    2. Candidate sampling includes hard positive and hard negative, each with k samples
    """

    # Get all unique IDs
    unique_ids = labels.unique()
    num_val_ids = int(len(unique_ids) * val_id_ratio)
    
    # Randomly select validation set IDs
    val_ids = set(random.sample(unique_ids.tolist(), num_val_ids))
    train_ids = set(unique_ids.tolist()) - val_ids
    
    # Store training and validation set results separately
    train_results = []
    val_results = []
    
    similarity_matrix = torch.mm(reid_embeds, reid_embeds.t())
    
    for i in range(similarity_matrix.size(0)):
        query_image_path = image_paths[i]
        query_label = labels[i]
        
        # Get indices of same-class samples
        same_class_mask = (labels == query_label)
        same_class_mask[i] = False  # Exclude self
        same_class_similarity = similarity_matrix[i][same_class_mask]
        same_class_indices = torch.where(same_class_mask)[0]
        
        # Get indices of different-class samples
        diff_class_mask = (labels != query_label)
        diff_class_similarity = similarity_matrix[i][diff_class_mask]
        diff_class_indices = torch.where(diff_class_mask)[0]
        
        # Get top k least similar samples from same class (hard positive)
        _, hard_pos_k_indices = torch.topk(same_class_similarity, min(k, len(same_class_similarity)), dim=0, largest=False)
        
        # Get top k most similar samples from different class (hard negative)
        _, hard_neg_k_indices = torch.topk(diff_class_similarity, min(k, len(diff_class_similarity)), dim=0, largest=True)
        
        # Process same-class samples (hard positive)
        for j in range(min(k, len(hard_pos_k_indices))):
            candidate_index = same_class_indices[hard_pos_k_indices[j]].item()
            candidate_image_path = image_paths[candidate_index]
            result = {"query": query_image_path, "candidate": candidate_image_path, "match": "Yes"}
            if query_label.item() in val_ids:
                val_results.append(result)
            else:
                train_results.append(result)
        
        # Process different-class samples (hard negative)
        for j in range(min(k, len(hard_neg_k_indices))):
            candidate_index = diff_class_indices[hard_neg_k_indices[j]].item()
            candidate_image_path = image_paths[candidate_index]
            result = {"query": query_image_path, "candidate": candidate_image_path, "match": "No"}
            if query_label.item() in val_ids:
                val_results.append(result)
            else:
                train_results.append(result)

    return train_results, val_results, train_ids, val_ids

def create_top_k_similar_pairwise_dataset_enhanced_testset(query_embeds, gallery_embeds, query_paths, gallery_paths, query_labels, gallery_labels, k, val_id_ratio):
    # Get all unique IDs
    unique_query_ids = query_labels.unique()
    num_test_ids = int(len(unique_query_ids) * val_id_ratio)
    
    # Randomly select test set IDs
    test_ids = set(random.sample(unique_query_ids.tolist(), num_test_ids))
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(query_embeds, gallery_embeds.t())
    
    # Build validation set
    test_set = []
    for i in range(query_embeds.size(0)):
        query_path = query_paths[i]
        current_label = query_labels[i]
        
        # Skip if current query's label is not in test set
        if current_label.item() not in test_ids:
            continue
        
        # Get similarity vector for current query
        sim_vector = similarity_matrix[i]
        
        # Get indices of positive and negative samples
        positive_indices = (gallery_labels == current_label).nonzero(as_tuple=True)[0]
        negative_indices = (gallery_labels != current_label).nonzero(as_tuple=True)[0]
        
        # Get easy positive and hard positive
        positive_sim = sim_vector[positive_indices]
        easy_positive_indices = positive_indices[torch.topk(positive_sim, min(k, len(positive_sim)), largest=True).indices]
        hard_positive_indices = positive_indices[torch.topk(positive_sim, min(k, len(positive_sim)), largest=False).indices]
        
        # Get easy negative and hard negative
        negative_sim = sim_vector[negative_indices]
        easy_negative_indices = negative_indices[torch.topk(negative_sim, min(k, len(negative_sim)), largest=False).indices]
        hard_negative_indices = negative_indices[torch.topk(negative_sim, min(k, len(negative_sim)), largest=True).indices]
        
        # Process same-class samples (hard positive)
        for idx in hard_positive_indices:
            candidate_path = gallery_paths[idx]
            test_set.append({"query": query_path, "candidate": candidate_path, "match": "Yes"})
        
        # Process same-class samples (easy positive)
        for idx in easy_positive_indices:
            candidate_path = gallery_paths[idx]
            test_set.append({"query": query_path, "candidate": candidate_path, "match": "Yes"})
            
        # Process different-class samples (hard negative)
        for idx in hard_negative_indices:
            candidate_path = gallery_paths[idx]
            test_set.append({"query": query_path, "candidate": candidate_path, "match": "No"})
            
        # Process different-class samples (easy negative)
        for idx in easy_negative_indices:
            candidate_path = gallery_paths[idx]
            test_set.append({"query": query_path, "candidate": candidate_path, "match": "No"})
    
    return test_set, test_ids

# Example usage
# similarity_matrix = ... # Computed similarity matrix
# reid_embeds = ... # ReID embedding vectors
# image_paths = ... # List of image paths
# labels = ... # List of image labels
# k = 5 # top k
# output_json_path = "/path/to/output.json"
# find_top_k_similar(similarity_matrix, reid_embeds, image_paths, labels, k, output_json_path)