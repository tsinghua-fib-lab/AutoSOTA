import gc
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Dict
import open_clip.transformer
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from dotmap import DotMap
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_r_precision
from tqdm import tqdm
import sys 
from data_utils import PROJECT_ROOT, get_dataset, RETRIEVAL_SPLTS
from datasets import DASSL_DATASETS, NANOBEIR_DATASETS
from datasets.roxford_rparis import ROxfordRParisDataset, compute_map
from utils import get_features, load_clip 
from utils import retrieval_average_precision_atk
from encode_no_projection import get_projection_layers
from utilities_summary import get_launcher_name, clean_metric_keys, flatten_args 
import os
import sys
import csv
import random
import string
import datetime


if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
base_path = Path(__file__).absolute().parents[1].absolute()


def apply_iso(W_text, W_image, iso_ktop=0, iso_kbottom=0, iso_tau=0.0):

        # inter-modal operator
        Psi = W_image.T @ W_text

        U, S, V = torch.linalg.svd(Psi, full_matrices=False)
        V = V.T

        r = S.shape[0]

        # Sanity check
        if iso_ktop + iso_kbottom >= r:
            raise ValueError(f"Cannot remove {iso_ktop} top and {iso_kbottom} bottom components from {r}")

        if iso_tau > 0:
            # Soft sigmoid-weighted spectral thresholding
            print("Soft filtering: k_top = {}, k_bottom = {}, tau = {}".format(iso_ktop, iso_kbottom, iso_tau))
            indices = torch.arange(r, device=S.device, dtype=S.dtype)
            w_top = torch.sigmoid((indices - iso_ktop) / iso_tau)
            w_bottom = torch.sigmoid((r - iso_kbottom - 1 - indices) / iso_tau)
            weights = w_top * w_bottom
            W_text_iso = W_text @ V @ (weights.unsqueeze(1) * V.T)
            W_image_iso = W_image @ U @ (weights.unsqueeze(1) * U.T)
        else:
            # Original hard binary selection
            print("Manual filtering: k_top = {}, k_bottom = {}".format(iso_ktop, iso_kbottom))
            start = iso_ktop
            end = r - iso_kbottom
            U_k = U[:, start:end]
            V_k = V[:, start:end]
            W_text_iso = W_text @ V_k @ V_k.T
            W_image_iso = W_image @ U_k @ U_k.T

        # Perform the transpose to match the original shape of the features
        W_text_iso = W_text_iso.T
        W_image_iso = W_image_iso.T

        return W_text_iso, W_image_iso
@torch.no_grad()
def compute_retrieval(gallery_eval_type: str, query_eval_type: str, dataroot: Path, dataset_name: str,
                      clip_model_name: str, use_open_clip: bool = False, open_clip_pretrained: str = None, **kwargs) -> Dict[str, float]:
    """
    Compute retrieval metrics based on the specified model, modalities and dataset.

    Args:
        gallery_eval_type (str): The type of evaluation for the gallery set (e.g., 'image', 'text', 'oti', 'ovi').
        query_eval_type (str): The type of evaluation for the query set (e.g., 'image', 'text', 'oti', 'ovi').
        dataroot (Path): The path to the dataset.
        dataset_name (str): The name of the dataset.
        clip_model_name (str): The name of the CLIP model to use.
        use_open_clip (bool, optional): Whether to use OpenCLIP. Defaults to False.
        open_clip_pretrained (str, optional): The pretrained model name for OpenCLIP. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Dict[str, float]: A dictionary containing the computed retrieval metrics for the dataset.

    Raises:
        ValueError: If the `query_eval_type` or `gallery_eval_type` are unknown.
    """
    clip_model, clip_model_name, clip_preprocess = load_clip(clip_model_name, open_clip_pretrained, use_open_clip,
                                                             device)
    
    # ISO Parameters 
    iso_ktop = kwargs.get('iso_ktop', 0)  # Middle components to select, 0 means no middle components
    iso_kbottom = kwargs.get('iso_kbottom', 0)  # Bottom components to select, 0 means no bottom components   
 

    query_split = kwargs.get("query_split", RETRIEVAL_SPLTS[dataset_name]["query"])
    gallery_split = kwargs.get("gallery_split", RETRIEVAL_SPLTS[dataset_name]["gallery"])

    use_iso = not kwargs.get("no_iso", False) 
    
    if use_iso:
        if (query_eval_type == "image" and gallery_eval_type == "text") or (query_eval_type == "text" and gallery_eval_type == "image"):
            sys.exit("ISO projection is not applicable for cross-modal retrieval. Please disable ISO or choose compatible evaluation types.")
     
    
    query_features_type = query_eval_type 
    gallery_features_type = gallery_eval_type 


    # Get gallery, query features and labels
    print("\nExtracting Gallery")
    gallery_features, gallery_labels = get_retrieval_features(dataroot, gallery_eval_type,  
                                                              clip_model_name,
                                                              dataset_name, gallery_split,
                                                              features_type=gallery_features_type,
                                                              use_open_clip=use_open_clip,
                                                              open_clip_pretrained=open_clip_pretrained,
                                                              use_iso=use_iso,
                                                              )
    print("\nExtracting Queries")
    query_features, query_labels = get_retrieval_features(dataroot, query_eval_type, 
                                                          clip_model_name,
                                                          dataset_name, query_split,
                                                          features_type=query_features_type,
                                                          use_open_clip=use_open_clip,
                                                          open_clip_pretrained=open_clip_pretrained,
                                                           use_iso=use_iso 
                                                          )

    gallery_features = gallery_features.float().to(device)
    query_features = query_features.float().to(device)
    
    if use_iso:
        # utility to extract projection layers from clip model
        W_image, W_text = get_projection_layers(clip_model , clip_model_name) 
        W_image = W_image.T 
        W_text = W_text.T 
        # Apply iso to text and image projectors
        iso_tau = kwargs.get('iso_tau', 0.0)
        use_iso_ensemble = kwargs.get('iso_ensemble', False)

        if use_iso_ensemble:
            band_configs = [(100, 25), (150, 50), (200, 75), (250, 100)]
            print(f"Multi-band ensemble with {len(band_configs)} configs: {band_configs}")

            all_similarities = []
            for bk_top, bk_bottom in band_configs:
                W_t_iso, W_i_iso = apply_iso(W_text, W_image, bk_top, bk_bottom, iso_tau)
                W_iso = W_i_iso if query_features_type == 'image' else W_t_iso
                q_proj = query_features @ W_iso
                g_proj = gallery_features @ W_iso
                q_proj = F.normalize(q_proj)
                g_proj = F.normalize(g_proj)
                sim = calculate_similarities(dataset_name, g_proj, q_proj, kwargs.get('split_size', 32))
                all_similarities.append(sim)

            similarities = torch.stack(all_similarities).mean(dim=0)

            if query_split == gallery_split and query_features_type == gallery_features_type:
                is_query_gallery_split_same = True
            else:
                is_query_gallery_split_same = False

            metrics = get_retrieval_metrics(dataset_name, similarities, query_labels, gallery_labels,
                                            is_query_gallery_split_same=is_query_gallery_split_same)
            gc.collect()
            torch.cuda.empty_cache()
            return {f'{dataset_name}_{key}': value for key, value in metrics.items()}

        W_text_iso, W_image_iso = apply_iso(W_text, W_image, iso_ktop, iso_kbottom, iso_tau)
    
        
        if query_features_type == "image" and gallery_features_type == "image":
            
            if isinstance(clip_model.visual, open_clip.timm_model.TimmModel): 
                # This is the case of Perception Encoder, EVA02 and Siglip-v2
                if clip_model_name == 'PE-Core-B-16-meta' or "EVA" in clip_model_name:
                    print("Not add ones for bias term for PE-Core-B-16-meta and EVA02-CLIP")
                else:
                    print("Adding ones for bias term for Siglip-v2")
                    ones_q = query_features.new_ones(query_features.size(0), 1)   # [N_q × 1]
                    ones_g = gallery_features.new_ones(gallery_features.size(0), 1)  # [N_g × 1]
                    query_features = torch.cat([query_features, ones_q], dim=1)     # [N_q × (d_t+1)]
                    gallery_features = torch.cat([gallery_features, ones_g], dim=1) # [N_g × (768+1)]
                
            query_features = query_features @ W_image_iso
            gallery_features = gallery_features @ W_image_iso
        
        elif query_features_type == "text"  and gallery_features_type == "text":
            
            query_features = query_features @ W_text_iso
            gallery_features = gallery_features @ W_text_iso
    else:
        print("Standard zero-shot retrieval without ISO projection")

              
    # Compute the similarity matrices
    gallery_features = F.normalize(gallery_features)
    query_features = F.normalize(query_features)
    
    if query_split == gallery_split and query_features_type == gallery_features_type:
        is_query_gallery_split_same = True
    else:
        is_query_gallery_split_same = False
    
    # For very large datasets like iNaturalist2021, compute metrics on-the-fly to save memory
    if dataset_name == 'inaturalist2021':
        metrics = compute_metrics_memory_efficient(dataset_name, gallery_features, query_features, 
                                                   query_labels, gallery_labels, is_query_gallery_split_same)
    else:
        similarities = calculate_similarities(dataset_name, gallery_features, query_features, kwargs.get('split_size', 32))
        metrics = get_retrieval_metrics(dataset_name, similarities, query_labels, gallery_labels,
                                        is_query_gallery_split_same=is_query_gallery_split_same)

    gc.collect()
    torch.cuda.empty_cache()

    # Add the dataset name to each key of the `metrics` dict
    return {f'{dataset_name}_{key}': value for key, value in metrics.items()}


@torch.no_grad()
def compute_metrics_memory_efficient(dataset_name: str, gallery_features: torch.Tensor, query_features: torch.Tensor,
                                    query_labels: torch.Tensor, gallery_labels: torch.Tensor, 
                                    is_query_gallery_split_same: bool = False, chunk_size: int = 1000) -> Dict[str, float]:
    """
    Compute retrieval metrics without storing the full similarity matrix (memory-efficient for large datasets).
    
    Args:
        dataset_name (str): The dataset name.
        gallery_features (torch.Tensor): Normalized gallery features on GPU.
        query_features (torch.Tensor): Normalized query features on GPU.
        query_labels (torch.Tensor): The ground truth labels for the query set.
        gallery_labels (torch.Tensor): The ground truth labels for the gallery set.
        is_query_gallery_split_same (bool): Whether the query and gallery splits are the same.
        chunk_size (int): Number of queries to process at once.
    
    Returns:
        Dict[str, float]: A dictionary containing computed metrics.
    """
    num_queries = query_features.shape[0]
    
    aps = []
    aps_at_r = []
    precisions_at_r = []
    recall_at_1 = []
    
    # Process queries in chunks
    for chunk_start in tqdm(range(0, num_queries, chunk_size), desc='Computing retrieval metrics'):
        chunk_end = min(chunk_start + chunk_size, num_queries)
        
        # Compute similarities for this chunk of queries
        query_chunk = query_features[chunk_start:chunk_end]
        similarities_chunk = torch.matmul(query_chunk, gallery_features.T)  # Shape: [chunk_size, num_gallery]
        
        # Compute ground truth for this chunk only
        query_labels_chunk = query_labels[chunk_start:chunk_end]
        ground_truth_chunk = (query_labels_chunk.unsqueeze(1) == gallery_labels.unsqueeze(0))  # [chunk_size, num_gallery]
        
        # Process each query in the chunk
        for i in range(similarities_chunk.shape[0]):
            query_idx = chunk_start + i
            query_sim = similarities_chunk[i].to(device)
            query_true = ground_truth_chunk[i].to(device)
            relevant_per_query = torch.sum(query_true)
            
            if is_query_gallery_split_same:  # Remove the query image from the gallery set
                query_sim = torch.cat((query_sim[:query_idx], query_sim[query_idx + 1:]))
                query_true = torch.cat((query_true[:query_idx], query_true[query_idx + 1:]))
            
            # Get the top 5 most similar images
            top_5_similarities, top_5_indices = torch.topk(query_sim, 5)
            
            # mAP (mean Average Precision)
            ap = retrieval_average_precision(query_sim, query_true)
            aps.append(ap.item())
            
            # mAP at R (where R is the number of relevant images)
            ap_at_r = retrieval_average_precision_atk(query_sim, query_true, top_k=relevant_per_query)
            aps_at_r.append(ap_at_r.item())
            
            # Precision at R
            precision_at_r = retrieval_r_precision(query_sim, query_true)
            precisions_at_r.append(precision_at_r.item())
            
            # Recall at 1 (whether the most similar image is relevant)
            recall_at_1.append(query_true[top_5_indices[0]].int().item())
    
    # Compile the results into a dictionary
    return_dict = {
        'mAP': np.mean(aps) * 100,
        'mAP_at_R': np.mean(aps_at_r) * 100,
        'precision_at_R': np.mean(precisions_at_r) * 100,
        'recall_at_1': np.mean(recall_at_1) * 100
    }
    
    return return_dict


def calculate_similarities(dataset_name: str, gallery_features: torch.Tensor, query_features: torch.Tensor,
                           split_size: Optional[int] = 32) -> torch.Tensor:
    """
    Calculate the cosine similarity between gallery and query features.

    Args:
        dataset_name (str): The dataset name to handle special cases.
        gallery_features (torch.Tensor): The features of the gallery set.
        query_features (torch.Tensor): The features of the query set.
        split_size (Optional[int]): The batch size to split the query features for memory efficiency.

    Returns:
        torch.Tensor: A tensor of cosine similarities between the query and gallery features.
    """
    # Compute the cosine similarity in a batched manner to avoid memory issues
    if dataset_name in ['imagenet', 'inaturalist2021']:  # For imagenet and iNaturalist we use float16 to save CPU memory
        split_size = 16 
        similarities = torch.empty((query_features.shape[0], gallery_features.shape[0]), device='cpu',
                                   dtype=torch.float16)
        splitted_query_features = torch.split(query_features, split_size)
        with torch.cuda.amp.autocast():
            for i, query_batch in tqdm(enumerate(splitted_query_features), total=len(splitted_query_features),
                                       desc='Computing similarities'):
                start_idx = i * split_size
                end_idx = start_idx + query_batch.size(0)
                similarities[start_idx:end_idx] = torch.matmul(query_batch, gallery_features.T).cpu()
    else:
        similarities = torch.vstack([torch.matmul(query_feat, gallery_features.T).cpu()
                                     for query_feat in
                                     tqdm(query_features.split(split_size), desc="Computing similarities")])
    return similarities


@torch.no_grad()
def get_retrieval_features(dataroot: Path, eval_type: str,  clip_model_name: str, dataset_name: str, split: str,
                           use_open_clip: bool = False, open_clip_pretrained: str = None, use_iso: bool = False, **kwargs) -> torch.Tensor:
    """
    Retrieve features for a given evaluation type (image, text, oti, ovi), model, split and dataset.

    Args:
        dataroot (Path): The root path to the dataset.
        eval_type (str): The evaluation type ('image', 'text').
        names_list (List[str]): The list of names corresponding to the data.
        clip_model (CLIP): The CLIP model to use for feature extraction.
        clip_model_name (str): The name of the CLIP model.
        dataset_name (str): The dataset name.
        split (str): The dataset split ('train', 'test', etc.), depending on the dataset.
        use_open_clip (bool): Whether to use OpenCLIP. Defaults to False.
        open_clip_pretrained (str): The pretrained model name for OpenCLIP. Defaults to None.

    Returns:
        torch.Tensor: The features for the given evaluation type.
        torch.Tensor: The labels for the corresponding features.

    Raises:
        ValueError: If the `eval_type` is unknown.
    """
    features_type = kwargs.get('features_type', 'image')
 
    features, names = get_features(dataroot, dataset_name, split, clip_model_name, features_type,
                                    use_open_clip=use_open_clip,
                                    open_clip_pretrained=open_clip_pretrained, use_iso=use_iso, **kwargs)

    if eval_type in ['image', 'text']:
        dummy_dataset = get_dataset(dataroot, dataset_name, split, preprocess=lambda x: x,
                                    **kwargs)  # Dataset used only to get the labels
        labels = torch.tensor(dummy_dataset.get_labels(features_type=features_type))

    else:
        raise ValueError(f"Unknown eval type {eval_type}")

    return features, labels


def compute_cross_modal_metrics(similarities: torch.Tensor, query_labels: torch.Tensor,
                                index_labels: torch.Tensor, query_image_similarities: Optional[torch.Tensor] = None,
                                is_query_index_split_same: bool = False):
    num_queries, num_gallery = similarities.shape
    ground_truth_tensor = (query_labels.unsqueeze(1) == index_labels.unsqueeze(0))

    aps = []
    aps_at_r = []  # Average precision at R where R is the number of relevant images per query
    precisions_at_r = []
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []

    if query_image_similarities is not None:
        query_recalls_at_1 = []
        query_recalls_at_3 = []
        query_recalls_at_5 = []

    for query in tqdm(range(num_queries), desc='Computing retrieval metrics'):
        query_sim = similarities[query].to(device)
        query_true = ground_truth_tensor[query].to(device)
        relevant_per_query = torch.sum(query_true)

        # Commented because it is impossible that this happen if it is cross-modal
        # if is_query_index_split_same:  # Remove the query image from the index set
        #     query_sim = torch.cat((query_sim[:query], query_sim[query + 1:]))
        #     query_true = torch.cat((query_true[:query], query_true[query + 1:]))

        top_10_similarities, top_10_indices = torch.topk(query_sim, 10)

        # # mAP
        ap = retrieval_average_precision(query_sim, query_true)
        aps.append(ap.item())
        #
        # # mAP at R
        ap_at_r = retrieval_average_precision_atk(query_sim, query_true, top_k=relevant_per_query)
        aps_at_r.append(ap_at_r.item())
        #
        # # Precision at R
        precision_at_r = retrieval_r_precision(query_sim, query_true)
        precisions_at_r.append(precision_at_r.item())

        # Recall at 1
        recall_at_1.append(query_true[top_10_indices[0]].int().item())

        # Recall at 5
        recall_at_5.append(torch.sum(query_true[top_10_indices[:5]]).clamp(0, 1).int().item())

        # Recall at 10
        recall_at_10.append(torch.sum(query_true[top_10_indices]).clamp(0, 1).int().item())

        
        if query_image_similarities is not None:
            query_im_sim = query_image_similarities[query].to(device)
            query_retrieval_pos = 10 - torch.searchsorted(top_10_similarities, query_im_sim,
                                                          sorter=torch.arange(len(top_10_similarities) - 1, -1, -1).to(
                                                              device), side='right')
            query_recalls_at_1.append(torch.sum(query_retrieval_pos < 1).item())
            query_recalls_at_3.append(torch.sum(query_retrieval_pos < 3).item())
            query_recalls_at_5.append(torch.sum(query_retrieval_pos < 5).item())

    return_dict = {
        'mAP': np.mean(aps) * 100,
        'mAP_at_R': np.mean(aps_at_r) * 100,
        'precision_at_R': np.mean(precisions_at_r) * 100,
        'recall_at_1': np.mean(recall_at_1) * 100,
        'recall_at_5': np.mean(recall_at_5) * 100,
        'recall_at_10': np.mean(recall_at_10) * 100
    }

    if query_image_similarities is not None:
        return_dict.update({
            'query_recall_at_1': np.mean(query_recalls_at_1) * 100,
            'query_recall_at_3': np.mean(query_recalls_at_3) * 100,
            'query_recall_at_5': np.mean(query_recalls_at_5) * 100
        })

    return return_dict


def get_retrieval_metrics(dataset_name: str, similarities: torch.Tensor, query_labels: torch.Tensor,
                          gallery_labels: torch.Tensor, is_query_gallery_split_same: bool = False, **kwargs) -> Dict[
    str, float]:
    """
    Get the retrieval metrics based on the dataset and the similarity matrix.

    Args:
        dataset_name (str): The dataset name to select the proper retrieval metrics.
        similarities (torch.Tensor): A tensor containing the similarity values between query and gallery items.
        query_labels (torch.Tensor): The labels for the query set.
        gallery_labels (torch.Tensor): The labels for the gallery set.
        is_query_gallery_split_same (bool): Whether the query and gallery splits are the same.

    Returns:
        Dict[str, float]: A dictionary containing retrieval metrics (e.g., mAP, precision, recall).

    Raises:
        ValueError: If the dataset is unknown.
    """
    if dataset_name in ['roxford5k', 'rparis6k']:
        return compute_roxford_rparis_metrics(dataset_name, similarities)
    elif dataset_name in ['coco_text', 'flickr30k_text', 'nocaps_text', 'cub2011', 'sop',
                          'inaturalist2021', 'places365',
                          'newsgroup_text', 'imdb_text'] + DASSL_DATASETS + NANOBEIR_DATASETS:
        return compute_metrics(dataset_name, similarities, query_labels, gallery_labels, is_query_gallery_split_same)
    
    elif dataset_name in ['coco', 'flickr30k']:
        return compute_cross_modal_metrics(similarities, query_labels, gallery_labels, None,
                                           is_query_gallery_split_same)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def compute_roxford_rparis_metrics(dataset_name: str, similarities: torch.Tensor) -> Dict[str, float]:
    """
    Computes retrieval metrics for the ROxford5k and RParis6k datasets. The metrics include
    mean Average Precision (mAP) for three cases (Easy, Medium, and Hard).

    Args:
        dataset_name (str): The name of the dataset ('roxford5k' or 'rparis6k').
        similarities (torch.Tensor): A tensor containing the computed similarities between queries and gallery items.

    Returns:
        Dict[str, float]: A dictionary containing the computed retrieval metrics (mAP for easy, medium, and hard).
    """
    similarities = similarities.cpu().numpy().T  # Transpose similarities for easier processing

    # Get ground truth for the dataset
    gnd = ROxfordRParisDataset.get_ground_truth(dataset_name)

    # Sort similarities in descending order to get rankings
    ranks = np.argsort(-similarities, axis=0)

    # Evaluation metrics at different top-K values
    ks = [1, 5, 10]

    # Search for easy images (only easy images are considered as relevant)
    gnd_t = []
    for i in range(len(gnd)):  # For each query
        g = {'ok': np.concatenate([gnd[i]['easy']]), 'junk': np.concatenate([gnd[i]['junk'], gnd[i]['hard']])}
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

    # Search for easy and hard images as relevant
    gnd_t = []
    for i in range(len(gnd)):
        g = {'ok': np.concatenate([gnd[i]['easy'], gnd[i]['hard']]), 'junk': np.concatenate([gnd[i]['junk']])}
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

    # Search for hard images (only hard images are considered as relevant)
    gnd_t = []
    for i in range(len(gnd)):
        g = {'ok': np.concatenate([gnd[i]['hard']]), 'junk': np.concatenate([gnd[i]['junk'], gnd[i]['easy']])}
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

    # Collect results in a dictionary
    return_dict = {
        'mAP_easy': mapE * 100,
        'mAP_medium': mapM * 100,
        'mAP_hard': mapH * 100
    }

    # Add precision at different top-k values for easy, medium, and hard cases
    for idx, k in enumerate(ks):
        return_dict.update({
            f'mP@{k}_easy': mprE[idx] * 100,
            f'mP@{k}_medium': mprM[idx] * 100,
            f'mP@{k}_hard': mprH[idx] * 100
        })

    return return_dict

def compute_metrics(dataset_name: str, similarities: torch.Tensor, query_labels: torch.Tensor,
                    gallery_labels: torch.Tensor, is_query_gallery_split_same: bool = False) -> Dict[str, float]:
    """
    Compute the retrieval metrics (e.g., mAP, Precision at R, Recall at 1) based on the dataset and similarities.

    Args:
        dataset_name (str): The dataset name (e.g., 'roxford5k', 'rparis6k').
        similarities (torch.Tensor): The similarity matrix between the query and gallery features.
        query_labels (torch.Tensor): The ground truth labels for the query set.
        gallery_labels (torch.Tensor): The ground truth labels for the gallery set.
        is_query_gallery_split_same (bool): Whether the query and gallery splits are the same.

    Returns:
        Dict[str, float]: A dictionary containing computed metrics (e.g., mAP, precision, recall).
    """
    num_queries, num_gallery = similarities.shape

    # Determine the ground truth based on the dataset
    if dataset_name in NANOBEIR_DATASETS:
        dummy_dataset = get_dataset('.', dataset_name, 'query', None)
        ground_truth_tensor = dummy_dataset.get_ground_truth()
    else:
        ground_truth_tensor = (query_labels.unsqueeze(1) == gallery_labels.unsqueeze(0))

    aps = []
    aps_at_r = []  # Average precision at R where R is the number of relevant images per query
    precisions_at_r = []
    recall_at_1 = []

    for query in tqdm(range(num_queries), desc='Computing retrieval metrics'):
        query_sim = similarities[query].to(device)
        query_true = ground_truth_tensor[query].to(device)
        relevant_per_query = torch.sum(query_true)

        if is_query_gallery_split_same:  # Remove the query image from the gallery set
            query_sim = torch.cat((query_sim[:query], query_sim[query + 1:]))
            query_true = torch.cat((query_true[:query], query_true[query + 1:]))

        # Get the top 5 most similar images
        top_5_similarities, top_5_indices = torch.topk(query_sim, 5)

        # mAP (mean Average Precision)
        ap = retrieval_average_precision(query_sim, query_true)
        aps.append(ap.item())

        # mAP at R (where R is the number of relevant images)
        ap_at_r = retrieval_average_precision_atk(query_sim, query_true, top_k=relevant_per_query)
        aps_at_r.append(ap_at_r.item())

        # Precision at R
        precision_at_r = retrieval_r_precision(query_sim, query_true)
        precisions_at_r.append(precision_at_r.item())

        # Recall at 1 (whether the most similar image is relevant)
        recall_at_1.append(query_true[top_5_indices[0]].int().item())

    # Compile the results into a dictionary
    return_dict = {
        'mAP': np.mean(aps) * 100,
        'mAP_at_R': np.mean(aps_at_r) * 100,
        'precision_at_R': np.mean(precisions_at_r) * 100,
        'recall_at_1': np.mean(recall_at_1) * 100
    }

    return return_dict


def init_retrieval_args(args: Namespace) -> DotMap:
    """
    Initialize the retrieval arguments by setting up the correct splits and loading required models.

    Args:
        args (Namespace): The parsed arguments from the command line.

    Returns:
        DotMap: The updated arguments as a DotMap.
    """
    if not isinstance(args, DotMap):
        args = DotMap(vars(args), _dynamic=False)

    # Set the query and gallery splits
    args.query_split = args.query_split if args.query_split is not None else RETRIEVAL_SPLTS[args.dataset_name]["query"]
    args.gallery_split = args.gallery_split if args.gallery_split is not None else RETRIEVAL_SPLTS[args.dataset_name]["gallery"]

    # Load the clip model and set the model name
    clip_model, clip_model_name, _ = load_clip(args.clip_model_name, args.open_clip_pretrained, args.use_open_clip, device)
    args.clip_model_name = clip_model_name


    # Set OpenCLIP flags and parameters
    args.use_open_clip = args.get('use_open_clip', False)
    args.open_clip_pretrained = args.get('open_clip_pretrained', None)

    return args



def add_args_to_parser() -> ArgumentParser:
    """
    Add arguments to the argument parser for the retrieval script.

    Returns:
        ArgumentParser: The argument parser instance with added arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--dataroot", required=True, help="Root directory containing all datasets.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate.")
    parser.add_argument("--clip_model_name", default="ViT-B/32", type=str, help="CLIP model variant to use, e.g. 'ViT-B/32'.")
    
    parser.add_argument("--query_eval_type", type=str,
                        choices=['image', 'text'],
                        required=True, help="Type of feature used for query:\n"
                                            "'image' for original image encoder features,\n"
                                            "'text' for original text encoder features,\n" )

    parser.add_argument("--gallery_eval_type", type=str,
                        choices=['image', 'text'],
                        required=True, help="Type of feature used for gallery:\n"
                                            "'image' for original image encoder features,\n"
                                            "'text' for original text encoder features,\n")
    
    parser.add_argument("--no_iso", action="store_true", default=False, help="Disable IsoCLIP.")
    
    parser.add_argument("--iso_ktop", type=int, default=150, help="Top K for ISO projection.")
    
    parser.add_argument("--iso_kbottom", type=int, default=50, help="Bottom K for ISO projection.")

    parser.add_argument("--iso_tau", type=float, default=0.0, help="Sigmoid temperature for soft spectral thresholding (0=hard).")

    parser.add_argument("--iso_ensemble", action="store_true", default=False, help="Multi-band ensemble averaging.")


    parser.add_argument("--query_split", type=str, default=None,
                        help="Dataset split used for query samples (e.g., 'train', 'test').")
    parser.add_argument("--gallery_split", type=str, default=None,
                        help="Dataset split used for gallery samples (e.g., 'train', 'test').")

    parser.add_argument("--use_open_clip", action='store_true', help="Enable to use OpenCLIP instead of OpenAI CLIP.",
                        default=False)
    parser.add_argument("--open_clip_pretrained", type=str,
                        help="Name of the pretrained weights for OpenCLIP (e.g., 'laion2b_s34b_b79k').", default=None)
    
    parser.add_argument("--out_path", type=str, default=None, help="Path to save results")
 
 
    return parser


def main():
    parser = add_args_to_parser()
    args = parser.parse_args()
    row_data = flatten_args(args) 
    
    args = init_retrieval_args(args)
    # assign  query split and gallery split from init_retreival_args for the final summary
    row_data["query_split"] = args.query_split
    row_data["gallery_split"] = args.gallery_split

    use_iso = not args.no_iso
 
    if use_iso:
        print(f"ISO Top K = {args.iso_ktop}, ISO Bottom K = {args.iso_kbottom}")
    else:
        print("No ISO projection applied") 
        args.iso_ktop = -1 
        args.iso_kbottom = -1 
        row_data["iso_ktop"], row_data["iso_kbottom"] = -1, -1
        
            
    metrics = compute_retrieval(**args)
    
    # Print results
    print("\n\n")
    print(f"clip_model_name = {args.clip_model_name}")
    print(f"dataset = {args.dataset_name}")
    print(f"query_eval_type = {args.query_eval_type}")
    print(f"gallery_eval_type = {args.gallery_eval_type}")
    print(f"query_split = {args.query_split}")
    print(f"gallery_split = {args.gallery_split}")
 
    print(f"use_open_clip = {args.use_open_clip}")
    print(f"open_clip_pretrained = {args.open_clip_pretrained}")
    print("\n\n")

    if args.dataset_name in ['roxford5k', 'rparis6k']:
        metrics.update({"{}_mAP".format(args.dataset_name): metrics["{}_mAP_easy".format(args.dataset_name)]})
        
    # Output the results
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name} = {metric_value:.2f}")

    
    """
    Code for summarize the results in a folder with random name
    """
    
    if args.out_path is not None:
        print(args.out_path)
        base_dir = os.path.join("results", args.out_path)
   
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    else:
        # 🧩 Run directly (terminal, debugger, VSCode, etc.)
        if not os.path.exists("local_run_retrieval"):
            os.makedirs("local_run_retrieval")
            
        base_dir = "local_run_retrieval"
    
    # --- Unique run folder name ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_tag = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    run_dir = os.path.join(base_dir, f"exp_{timestamp}_{rand_tag}")
    os.makedirs(run_dir, exist_ok=True)

    
    metrics_clean = clean_metric_keys(metrics, args.dataset_name)
 
    if (args.query_eval_type == "image" and args.gallery_eval_type == "image") or (args.query_eval_type == "text" and args.gallery_eval_type == "text"): 
        metric_keys_master = [
                            # Generic metrics
                            "mAP", "mAP_at_R", "precision_at_R", "recall_at_1",
                            # Oxford/Paris style
                            "mAP_easy", "mAP_medium", "mAP_hard",
                            "mP@1_easy", "mP@1_medium", "mP@1_hard",
                            "mP@5_easy", "mP@5_medium", "mP@5_hard",
                            "mP@10_easy", "mP@10_medium", "mP@10_hard",
                        ]
    else:
        metric_keys_master = [
                    # Generic metrics for text-to-image
                    "mAP", "mAP_at_R", "precision_at_R", "recall_at_1", "recall_at_5", "recall_at_10" 
                    ]
        
    
    # --- Merge args + metrics into a single dict ---
 
    for key in metric_keys_master:
        row_data[key] = metrics_clean.get(key, None)
    row_data["folder_path"] = os.path.abspath(run_dir)
    row_data["timestamp"] = timestamp

    # --- Write CSV summary for this run ---
    csv_path = os.path.join(run_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        writer.writeheader()
        writer.writerow(row_data)

    print(f"\n✅ Summary saved at: {csv_path}\n")

 

if __name__ == '__main__':
    main()
