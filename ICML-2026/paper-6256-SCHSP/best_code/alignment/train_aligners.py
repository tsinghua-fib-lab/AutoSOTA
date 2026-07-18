#!/usr/bin/env python3
"""
Train an MLP to align ImageNet-1k class names to text embeddings from various text encoders.
The MLP learns a mapping from class name indices to text embeddings.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from datetime import datetime

# Import utility functions
from utils.utils import get_label_names, get_text_encoder, get_backbone, get_preprocess, get_one_caption_per_dataset, compute_and_save_mean_representation, apply_attention_based_preprocessing, apply_linear_projection_preprocessing, train_linear_projection
from analyses.modality_gap import load_image_representations_imagenet1k
from utils.combined_datasets import make_combined_loader
from utils.model_utils import load_head_weights
from dataloaders.datasets_and_dataloaders import get_dataloaders
from config.config import IMAGENET1K_HEAD_MODELS, IMAGENET21K_HEAD_MODELS, IMAGENET21K_2EXTRA_HEAD_MODELS, INATURALIST_HEAD_MODELS, EMBEDDINGS_DIR, WEIGHTS_DIR
from alignment.aligned_models import TextToImageEmbeddingDataset, TextToImageMLP, get_mlp_aligner_path
# Import CCA class
from other_methods.csa.cca_class import NormalizedCCA

from PIL import Image
import numpy as np


def get_text_embeddings(dataset_img_repr, text_model_name, device='cuda', use_captions=False):

    # print(f"Warning: Using captions. Harcoded Eliminate this line if not using captions.")
    if dataset_img_repr != "imagenet21k" and use_captions:
        class_names = get_one_caption_per_dataset(dataset_img_repr)
    else: # Use original class names for imagenet21k
        class_names = get_label_names(dataset_img_repr)
    all_prompts = class_names.copy()  # Start with original class names
    # all_prompts = ['A photo of ' + name.replace('_', ' ') for name in all_prompts]  # Remove empty prompts

    print(f"Total prompts for text embedding: {len(all_prompts)}")
    print("Generating text embeddings for class names...")
    text_encoder = get_text_encoder(text_model_name, device)
    batch_size = 5000
    text_features_list = []
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i + batch_size]
        batch_features = text_encoder(batch_prompts)
        text_features_list.append(batch_features)
        print(f"Processed batch {i // batch_size + 1}/{(len(all_prompts) + batch_size - 1) // batch_size}")

    text_embeddings = torch.cat(text_features_list, dim=0)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)  # Normalize text embeddings
    print(f"Text embeddings shape: {text_embeddings.shape}")
    return text_embeddings, len(class_names)

def get_avg_image_embeddings(dataset_img_repr, image_model_name, num_classes, device='cuda', few_shot_samples=None, only_test=False):
    """
    Generate image embeddings for a given dataset and model.
    
    Args:
        dataset_img_repr: Dataset name for image representation
        image_model_name: Name of the image model
        num_classes: Number of classes in the dataset
        device: Device to use for computation
        few_shot_samples: Number of samples per class to use (1, 2, 4, 8, or 16). If None, uses all samples.
    """
    # Compute image features using backbone
    print(f"Loading image backbone for {image_model_name}...")
    backbone, image_embedding_dim, _ = get_backbone(image_model_name)
    backbone.to(device)
    backbone.eval()

    print("Getting preprocessing transforms...")
    from torchvision import transforms
    preprocess = get_preprocess(image_model_name)
#     preprocess = transforms.Compose([
#     transforms.Resize(256, interpolation=Image.BICUBIC),
#     transforms.CenterCrop(224),
#     transforms.ToTensor()            
# ]) #to save the images without normalization

    print(f"Loading {dataset_img_repr} dataset for image features...")
    train_loader, val_loader, test_loader = get_dataloaders(dataset_img_repr, 128, preprocess, only_test)
    image_loader = train_loader if not only_test else test_loader
    print(f"Computing image features for all images in {dataset_img_repr}...")
   
    class_features = {i: [] for i in range(num_classes)}
    class_sample_counts = {i: 0 for i in range(num_classes)}
    saved_images = set()
    with torch.no_grad():
        for images, labels in tqdm(image_loader, desc="Extracting image features"):
            images = images.to(device)
            labels = labels.to(device)
            feats = backbone(images)
            # feats = F.normalize(feats, p=2, dim=-1)
            for idx, (feat, label) in enumerate(zip(feats, labels)):
                label_item = label.item()
                # If few_shot_samples is specified, only collect up to that many samples per class
                if few_shot_samples is None or class_sample_counts[label_item] < few_shot_samples:
                    class_features[label_item].append(feat.cpu())
                    class_sample_counts[label_item] += 1
                    # Save the first image for each class
                    # if label_item not in saved_images:
                    #     image_np = images[idx].cpu().permute(1, 2, 0).numpy()
                    #     image_np = (image_np * 255).astype(np.uint8)
                    #     save_dir = f"./captioned_images/{dataset_img_repr}"
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     print(f"Saving example image for class {label_item} to {save_dir}/{label_item}.jpg")
                    #     Image.fromarray(image_np).save(f"{save_dir}/{label_item}.jpg")
                    #     saved_images.add(label_item)
            # Early termination if we have enough samples for all classes
            if few_shot_samples is not None and all(count >= few_shot_samples for count in class_sample_counts.values()):
                print(f"✅ Collected {few_shot_samples} samples for all {num_classes} classes")
                break

    image_weights = torch.stack([
        torch.stack(class_features[i]).mean(dim=0) if len(class_features[i]) > 0 else torch.zeros(image_embedding_dim)
        for i in range(num_classes)
    ])
    # Ensure image_weights is normalized
    image_weights = F.normalize(image_weights, p=2, dim=-1)
    image_weights = image_weights.to(device)
    # Substract the mean
    # image_weights = image_weights - image_weights.mean(dim=0, keepdim=True)
    
    if few_shot_samples is not None:
        print(f"Computed image_weights from backbone with few-shot ({few_shot_samples} samples/class): {image_weights.shape}")
        # Report actual sample counts per class
        actual_counts = [len(class_features[i]) for i in range(num_classes)]
        print(f"Sample counts per class: min={min(actual_counts)}, max={max(actual_counts)}, avg={sum(actual_counts)/len(actual_counts):.1f}")
    else:
        print(f"Computed image_weights from backbone: {image_weights.shape}")
    
    return image_weights, image_embedding_dim

def get_image_embeddings(dataset_img_repr, split, batch_size, image_model_name, device='cuda', few_shot_samples=None):
    # Compute image features using backbone
    print(f"Loading image backbone for {image_model_name}...")
    backbone, image_embedding_dim, _ = get_backbone(image_model_name)
    backbone.to(device)
    backbone.eval()

    print("Getting preprocessing transforms...")
    preprocess = get_preprocess(image_model_name)

    print(f"Loading {dataset_img_repr} dataset for image features...")
    train_loader, val_loader, test_loader = get_dataloaders(dataset_img_repr, batch_size, preprocess, only_test=True if split=='test' else False,
                                                            valid_split=True if split=='val' else False, shuffle=False)
    if split == 'train':
        image_loader = train_loader
    elif split == 'val':
        image_loader = val_loader
    elif split == 'test':
        image_loader = test_loader
    else:
        raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'.")
    
    features = []
    total_samples = 0
    with torch.no_grad():
        for images, _ in tqdm(image_loader, desc="Extracting image features"):
            images = images.to(device)
            feats = backbone(images)
            feats = F.normalize(feats, p=2, dim=-1)
            features.append(feats)
            total_samples += feats.size(0)
            
            # Stop the loop if we have collected enough samples
            if few_shot_samples is not None and total_samples >= few_shot_samples:
                break
            
    features = torch.cat(features, dim=0)
    
    # Apply a mask to keep only the first few_shot_samples features
    if few_shot_samples is not None and few_shot_samples < features.size(0):
        features = features[:few_shot_samples]
    print(f"Extracted image features: {features.shape}")
    return features

def get_image_and_caption_embeddings(dataset_img_repr, split, batch_size, image_model_name, text_model_name, device='cuda', few_shot_samples=None):
    img_embeddings_name = f"{dataset_img_repr}_{split}_{str(few_shot_samples)}_{image_model_name.replace('/', '_')}.pt"
    text_embeddings_name = f"{dataset_img_repr}_{split}_{str(few_shot_samples)}_{text_model_name.replace('/', '_')}.pt"
    img_embeddings_path = os.path.join(EMBEDDINGS_DIR, img_embeddings_name)
    text_embeddings_path = os.path.join(EMBEDDINGS_DIR, text_embeddings_name)
    # Check if the embeddings exist
    if os.path.exists(img_embeddings_path) and os.path.exists(text_embeddings_path):
        print(f"Loading embeddings from {img_embeddings_name}...")
        image_features, text_features = torch.load(img_embeddings_path, map_location=device), torch.load(text_embeddings_path, map_location=device)
        return image_features, text_features, image_features.shape[1], text_features.shape[1]
    else:
        print(f"Embeddings not found, computing from scratch.")

    # Compute image features using backbone
    print(f"Loading image backbone for {image_model_name}...")
    backbone, image_embedding_dim, _ = get_backbone(image_model_name)
    backbone.to(device)
    backbone.eval()

    # Load text encoder for captions
    print(f"Loading text encoder for {text_model_name}...")
    text_encoder = get_text_encoder(text_model_name, device)

    print("Getting preprocessing transforms...")
    preprocess = get_preprocess(image_model_name)

    print(f"Loading {dataset_img_repr} dataset for image features...")
    train_loader, val_loader, test_loader = get_dataloaders(dataset_img_repr, batch_size, preprocess, only_test=True if split == 'test' else False,
                                                            valid_split=True if split == 'val' else False, shuffle=False)
    if split == 'train':
        image_loader = train_loader
    elif split == 'val':
        image_loader = val_loader
    elif split == 'test':
        image_loader = test_loader
    else:
        raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'.")

    image_features = []
    text_features = []
    total_samples = 0
    with torch.no_grad():
        for images, captions in tqdm(image_loader, desc="Extracting image and caption features"):
            images = images.to(device)
            feats = backbone(images)
            feats = F.normalize(feats, p=2, dim=-1)
            # Repeat image features 5 times for each caption
            repeated_feats = feats.repeat_interleave(5, dim=0)
            image_features.append(repeated_feats)

            # Process captions
            caption_features = []
            for caption in captions:
                # Process each element in the caption tuple
                caption_feats = [text_encoder([caption_elem]).to(device) for caption_elem in caption]
                # Normalize each feature and concatenate along the initial dimension
                caption_feat = torch.cat([F.normalize(caption_feat, p=2, dim=-1) for caption_feat in caption_feats], dim=0)
                caption_features.append(caption_feat)

            # Concatenate caption features
            caption_features = torch.cat(caption_features, dim=0)
            text_features.append(caption_features)

            total_samples += feats.shape[0] * 5  # Each image has 5 captions
            if few_shot_samples is not None and total_samples >= few_shot_samples * 5:
                break

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    if few_shot_samples is not None and few_shot_samples * 5 <= total_samples:
        # Apply mask to return only few_shot_samples * 5 representations
        mask = torch.arange(few_shot_samples * 5)
        image_features = image_features[mask]
        text_features = text_features[mask]

    image_embedding_dim = image_features.shape[1]
    text_embedding_dim = text_features.shape[1]

    # Ensure both lists have the same size
    assert image_features.shape[0] == text_features.shape[0], (
        f"Image and text features size mismatch: {image_features.shape[0]} vs {text_features.shape[0]}"
    )
    print(f"Extracted image features: {image_features.shape}")
    print(f"Extracted text features: {text_features.shape}")
    # Save the embeddings for future use
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    print(f"Saving embeddings to {img_embeddings_name} and {text_embeddings_name}...")
    torch.save(image_features, img_embeddings_path)
    torch.save(text_features, text_embeddings_path)
    return image_features, text_features, image_embedding_dim, text_embedding_dim


def get_img_repr_dataset(dataset_img_repr, text_model_name, image_model_name, device='cuda',
                            batch_size=64, few_shot_samples=None, use_captions=False):
    """
    Create a dataset for text-to-image embedding alignment.
    
    Args:
        dataset_img_repr: Dataset name for image representation
        text_model_name: Name of the text model
        image_model_name: Name of the image model
        device: Device to use for computation
        batch_size: Batch size (unused in this function but kept for compatibility)
        few_shot_samples: Number of samples per class to use (1, 2, 4, 8, or 16). If None, uses all samples.
        no_weights: If True, only use image representation datasets (no weights dataset). Requires dataset_img_repr to be not None.
    """
    if dataset_img_repr=='flickr30k':
        image_embeddings, text_embeddings,\
        image_embedding_dim, text_embedding_dim = get_image_and_caption_embeddings(
            dataset_img_repr=dataset_img_repr,
            split='train',  # Use train split for features
            batch_size=batch_size,
            image_model_name=image_model_name,
            text_model_name=text_model_name,
            device=device,
            few_shot_samples=few_shot_samples
        )
    else:
        if few_shot_samples != 1 and use_captions == True:
            print(f"⚠️ Warning: few_shot_samples is {few_shot_samples} but use_captions is True. Captions are only supported for few_shot_samples=1. Setting use_captions to False.")
            use_captions = False
        text_embeddings, num_classes = get_text_embeddings(dataset_img_repr,
                                            text_model_name, device, use_captions)
        text_embedding_dim = text_embeddings.shape[1]

        image_embeddings, image_embedding_dim = get_avg_image_embeddings(dataset_img_repr, image_model_name, num_classes, device, few_shot_samples)

    # Ensure text embeddings are not normalized and image embeddings are normalized
    text_normalized = torch.allclose(text_embeddings.norm(dim=-1), torch.tensor(1.0, device=device, dtype=text_embeddings.dtype), atol=1e-3)
    image_normalized = torch.allclose(image_embeddings.norm(dim=-1), torch.tensor(1.0, device=device, dtype=image_embeddings.dtype), atol=1e-1)
    if not text_normalized:
        raise ValueError("Text embeddings must be normalized. Please check the text model weights.")
    if not image_normalized:
        raise ValueError("Image embeddings must be normalized. Please check the image model weights.")
    # Create dataset and dataloader
    dataset = TextToImageEmbeddingDataset(text_embeddings, image_embeddings)
    print(f"Dataset created with {len(dataset)} samples")
    return dataset, text_embedding_dim, image_embedding_dim

def get_weights_repr(image_model_name, device='cuda', preprocess=None):
    """
    Get the weight representations for a given image model.
    
    Args:
        image_model_name: Name of the image model
        device: Device to use for computation
        preprocess: Preprocessing mode. Can be 'mean', 'attention', or 'linear'.
    
    Returns:
        weight_vectors: Tensor of shape (num_classes, embedding_dim)
    """
    # Load image model weights
    if image_model_name in IMAGENET1K_HEAD_MODELS:
        imagenet_pth_name = 'imagenet1k.pth'
    elif image_model_name in IMAGENET21K_HEAD_MODELS:
        imagenet_pth_name = 'imagenet21k.pth'
    elif image_model_name in IMAGENET21K_2EXTRA_HEAD_MODELS:
        imagenet_pth_name = 'imagenet21k_2extra.pth'
    elif image_model_name in INATURALIST_HEAD_MODELS:
        imagenet_pth_name = 'inaturalist.pth'
    else:
        raise ValueError(f"Unsupported image model: {image_model_name}. "
                         f"Supported models are: {IMAGENET1K_HEAD_MODELS + IMAGENET21K_HEAD_MODELS + IMAGENET21K_2EXTRA_HEAD_MODELS}")

    # Validate preprocess argument
    if preprocess is not None:
        if image_model_name not in IMAGENET1K_HEAD_MODELS:
            raise ValueError(f"Preprocessing (preprocess='{preprocess}') is only available for IMAGENET1K_HEAD_MODELS, "
                           f"but got {image_model_name}")
        if preprocess not in ['mean', 'attention', 'linear']:
            raise ValueError(f"Unsupported preprocessing mode: '{preprocess}'. Only 'mean', 'attention', and 'linear' are supported.")

    weights_path = os.path.join(WEIGHTS_DIR, image_model_name.replace('/', '_'), imagenet_pth_name)
    if not os.path.exists(weights_path):
        print(f"Image model weights not found: {weights_path}. Downloading...")
        # Call the function to download the weights
        load_head_weights(model_name=image_model_name, weights_dir=WEIGHTS_DIR)
    
    print(f"Loading image model weights from: {weights_path}")
    weights_data = torch.load(weights_path, map_location=device)
    weight_vectors = weights_data['weight'].float()  # Ensure float32
    
    # Apply preprocessing if specified
    if preprocess == 'mean':
        print("🔄 Applying mean-based preprocessing to weight vectors...")
        
        # Load the mean of imagenet1kval representations
        mean_repr_name = f"imagenet1kval_reprmean_{image_model_name.replace('/', '_')}.pt"
        mean_repr_path = os.path.join(EMBEDDINGS_DIR, mean_repr_name)
        
        if not os.path.exists(mean_repr_path):
            print(f"⚠️  Mean representation file not found: {mean_repr_path}")
            print("🔄 Computing mean representation on-the-fly...")
            img_repr_mean, mean_repr_path = compute_and_save_mean_representation(image_model_name, device)
        else:
            print(f"Loading mean representation from: {mean_repr_path}")
            img_repr_mean = torch.load(mean_repr_path, map_location=device).float()
        
        # Calculate mean of weight vectors across dimension 0 (across classes)
        weight_vectors_mean = weight_vectors.mean(dim=0, keepdim=True)
        
        print(f"  Image representation mean shape: {img_repr_mean.shape}")
        print(f"  Weight vectors mean shape: {weight_vectors_mean.shape}")
        
        # Apply transformation: weight_vectors = normalize(weight_vectors - weight_vectors_mean + img_repr_mean)
        weight_vectors = weight_vectors - weight_vectors_mean + img_repr_mean
        print(f"  Applied transformation: weight_vectors - weight_vectors_mean + img_repr_mean")
        
        # Normalize weights (for 'mean' preprocessing)
        weight_vectors = F.normalize(weight_vectors, p=2, dim=-1)
    
    elif preprocess == 'attention':
        print("🔄 Applying attention-based preprocessing to weight vectors...")
        
        # Get the name for the imagenet1k average representations for this image model
        img_repr_avg_name = f"imagenet1kval_reprmean_{image_model_name.replace('/', '_')}.pt"
        img_repr_avg_path = os.path.join(EMBEDDINGS_DIR, img_repr_avg_name)
        if not os.path.exists(img_repr_avg_path):
            print(f"⚠️  ImageNet-1k average representation file not found: {img_repr_avg_path}")
            print("🔄 Computing ImageNet-1k average representations on-the-fly...")
            img_reprs_avg, _ = get_avg_image_embeddings('imagenet1kval', image_model_name, num_classes=1000, device=device, few_shot_samples=50, only_test=True)
        else:
            print(f"Loading ImageNet-1k average representations from: {img_repr_avg_path}")
            img_reprs_avg = torch.load(img_repr_avg_path, map_location=device).float()

        # Apply attention-based preprocessing
        weight_vectors = apply_attention_based_preprocessing(weight_vectors, img_reprs_avg, device)
        # Note: The function already normalizes
        
    elif preprocess == 'linear':
        print("🔄 Applying linear projection preprocessing to weight vectors...")
        
        # Get the name for the imagenet1k average representations for this image model
        img_repr_avg_name = f"imagenet1kval_repravg_{image_model_name.replace('/', '_')}.pt"
        img_repr_avg_path = os.path.join(EMBEDDINGS_DIR, img_repr_avg_name)
        if not os.path.exists(img_repr_avg_path):
            print(f"⚠️  ImageNet-1k average representation file not found: {img_repr_avg_path}")
            print("🔄 Computing ImageNet-1k average representations on-the-fly...")
            img_reprs_avg, _ = get_avg_image_embeddings('imagenet1kval', image_model_name, num_classes=1000, device=device, few_shot_samples=None, only_test=True)
            torch.save(img_reprs_avg, img_repr_avg_path)
        else:
            print(f"Loading ImageNet-1k average representations from: {img_repr_avg_path}")
            img_reprs_avg = torch.load(img_repr_avg_path, map_location=device).float()
        
        # Apply linear projection preprocessing
        weight_vectors, linear_layer = apply_linear_projection_preprocessing(
            weight_vectors, img_reprs_avg, 
            linear_layer=None,
            num_epochs=500,
            learning_rate=0.01,
            weight_decay=0,
            dropout_rate=0.6,
            device=device,
            train_projection=True
        )
        # Note: The function already normalizes
        
    else:
        # Normalize weights (for no preprocessing)
        weight_vectors = F.normalize(weight_vectors, p=2, dim=-1)
        
    return weight_vectors

def get_weights_repr_dataset(text_model_name, image_model_name, device='cuda', batch_size=None, preprocess=None):
    """
    Create a dataset for text-to-image embedding alignment.
    
    Args:
        text_model_name: Name of the text model
        image_model_name: Name of the image model
        device: Device to use for computation
        batch_size: Batch size for training
        preprocess: Preprocessing mode. Can be 'mean' to apply mean-based normalization.
                    Only available for IMAGENET1K_HEAD_MODELS.
    """
    
    # Determine dataset for labels
    if image_model_name in IMAGENET1K_HEAD_MODELS:
        dataset_for_labels = "imagenet1k"
    elif image_model_name in IMAGENET21K_HEAD_MODELS:
        dataset_for_labels = "imagenet21k"
    elif image_model_name in IMAGENET21K_2EXTRA_HEAD_MODELS:
        dataset_for_labels = "imagenet21k_2extra"
    elif image_model_name in INATURALIST_HEAD_MODELS:
        dataset_for_labels = "inaturalist"
    else:
        raise ValueError(f"Unsupported image model: {image_model_name}. "
                         f"Supported models are: {IMAGENET1K_HEAD_MODELS + IMAGENET21K_HEAD_MODELS + IMAGENET21K_2EXTRA_HEAD_MODELS}")

    text_embeddings, num_classes = get_text_embeddings(dataset_for_labels, text_model_name, device)
    text_embedding_dim = text_embeddings.shape[1]

    # Get weight representations
    weight_vectors = get_weights_repr(image_model_name, device, preprocess)
    image_embedding_dim = weight_vectors.shape[1]
    
    print(f"Image embedding dimension: {image_embedding_dim}")
    print(f"Image weights shape: {weight_vectors.shape}")
    print(f"Image weights dtype: {weight_vectors.dtype}")
    
    if preprocess == 'mean':
        print("📝 Using original text embeddings and mean-preprocessed image weights...")
    elif preprocess == 'attention':
        print("📝 Using original text embeddings and attention-preprocessed image weights...")
    elif preprocess == 'linear':
        print("📝 Using original text embeddings and linear-projection-preprocessed image weights...")
    else:
        print("📝 Using original text embeddings and image weights...")
    
    dataset = TextToImageEmbeddingDataset(text_embeddings, weight_vectors)
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Final dataset shapes: text={text_embeddings.shape}, image={weight_vectors.shape}")
    
    # Use full batch gradient descent if batch_size is None
    if batch_size is None:
        batch_size = len(dataset)
        print(f"Using full batch gradient descent with batch size: {batch_size}")

    return dataset, text_embedding_dim, image_embedding_dim

def get_all_datasets(text_model_name, image_model_name, dataset_img_repr, few_shot_samples, no_weights, device, use_captions=False, preprocess=None):

    # Handle multiple dataset_img_repr inputs
    if isinstance(dataset_img_repr, list):
        dataset_img_repr_list = dataset_img_repr
        print(f"📊 Multiple image representation datasets provided: {dataset_img_repr_list}")
    elif dataset_img_repr is not None:
        dataset_img_repr_list = [dataset_img_repr]
        print(f"📊 Single image representation dataset provided: {dataset_img_repr}")
    else:
        dataset_img_repr_list = None

    # Handle few_shot_samples argument
    if few_shot_samples is not None and dataset_img_repr_list is not None:
        if isinstance(few_shot_samples, list):
            if len(few_shot_samples) != len(dataset_img_repr_list):
                raise ValueError(f"few_shot_samples list length ({len(few_shot_samples)}) must match dataset_img_repr list length ({len(dataset_img_repr_list)})")
            few_shot_samples_list = few_shot_samples
        else:
            # Single value, replicate for all datasets
            few_shot_samples_list = [few_shot_samples] * len(dataset_img_repr_list)

        print(f"📊 Few-shot samples configuration: {few_shot_samples_list}")
    elif dataset_img_repr_list is not None:
        few_shot_samples_list = [None] * len(dataset_img_repr_list)
        print(f"📊 Using all available samples (no few-shot restriction)")

    # Create image representation datasets for each dataset
    image_repr_datasets = []
    text_embedding_dim = None
    image_embedding_dim = None
    
    if dataset_img_repr_list is not None:
        for i, dataset_name in enumerate(dataset_img_repr_list):
            current_few_shot = few_shot_samples_list[i]
            print(f"\n🔄 Creating image representation dataset {i+1}/{len(dataset_img_repr_list)} for dataset: {dataset_name}")
            if current_few_shot is not None:
                print(f"   Using few-shot mode: {current_few_shot} samples per class")

            img_dataset, text_emb_dim, img_emb_dim = get_img_repr_dataset(
                dataset_img_repr=dataset_name,
                text_model_name=text_model_name,
                image_model_name=image_model_name,
                device=device,
                few_shot_samples=current_few_shot,
                use_captions=use_captions
            )

            image_repr_datasets.append(img_dataset)

            # Set dimensions from the first dataset (all should be the same)
            if text_embedding_dim is None:
                text_embedding_dim = text_emb_dim
                image_embedding_dim = img_emb_dim
                print(f"✅ Set embedding dimensions from {dataset_name}: text={text_embedding_dim}, image={image_embedding_dim}")
            else:
                # Verify dimensions match across datasets
                if text_emb_dim != text_embedding_dim or img_emb_dim != image_embedding_dim:
                    raise ValueError(f"Embedding dimensions mismatch for dataset {dataset_name}. "
                                f"Expected text={text_embedding_dim}, image={image_embedding_dim}, "
                                f"but got text={text_emb_dim}, image={img_emb_dim}")
                print(f"✅ Verified embedding dimensions match for {dataset_name}")

        print(f"\n📦 Successfully created {len(image_repr_datasets)} image representation datasets")

    else:
        image_repr_datasets = None
        print(f"\n⚠️ No image representation datasets provided. Using weights dataset only.")

    # Create weights dataset
    if not no_weights:
        print(f"\n🔄 Creating weights representation dataset...")
        weights_dataset, weights_text_emb_dim, weights_img_emb_dim = get_weights_repr_dataset(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            device=device,
            preprocess=preprocess
        )
        
        # Verify weights dataloader dimensions match
        if dataset_img_repr_list is not None:
            if weights_text_emb_dim != text_embedding_dim or weights_img_emb_dim != image_embedding_dim:
                raise ValueError(f"Weights dataloader dimensions mismatch. "
                            f"Expected text={text_embedding_dim}, image={image_embedding_dim}, "
                            f"but got text={weights_text_emb_dim}, image={weights_img_emb_dim}")
            print(f"✅ Verified weights dataloader dimensions match")

        text_embedding_dim = weights_text_emb_dim
        image_embedding_dim = weights_img_emb_dim
    else:
        print(f"\n⚠️ No weights mode enabled. Skipping weights dataset creation.")
        weights_dataset = None

    # Combine all datasets (image representation datasets + weights dataset if not no_weights)
    if no_weights:
        if image_repr_datasets is None:
            raise ValueError("Cannot use no_weights=True without image representation datasets")
        all_datasets = image_repr_datasets
        print(f"\n🔗 Using only image representation datasets: {len(all_datasets)} datasets")
    elif image_repr_datasets is None:
        all_datasets = [weights_dataset]
        print(f"\n🔗 Using only weights dataset: {len(all_datasets)} dataset")
    else:
        all_datasets = image_repr_datasets + [weights_dataset]
        print(f"\n🔗 Combining {len(all_datasets)} datasets ({len(image_repr_datasets)} image repr + 1 weights)")

    # if batch_size is None:
    #     print("⚠️  Warning: batch_size is None. Using default batch_size=128 for combined loader")
    #     batch_size = 128

    # Check if all features in each dataset are normalized (element by element)
    # if len(all_datasets) >= 1:
    #     print(f"🔄 Checking normalization for all features in {len(all_datasets)} datasets...")
    #     for dataset_idx, dataset in enumerate(all_datasets):
    #         if isinstance(dataset, TextToImageEmbeddingDataset):
    #             for i in range(len(dataset)):
    #                 sample_text_emb, sample_image_emb = dataset[i]
    #                 sample_text_emb = sample_text_emb.float()
    #                 sample_image_emb = sample_image_emb.float()
    #                 if not torch.allclose(sample_image_emb.norm(), torch.tensor(1.0, dtype=torch.float32), atol=1e-3):
    #                     raise ValueError(f"Image embedding at index {i} in dataset {dataset_idx} is not normalized. Norm: {sample_image_emb.norm().item()}")
    #         else:
    #             print(f"Skipping normalization check for non-TextToImageEmbeddingDataset: {type(dataset)}")
    #     print("✅ All features in all datasets are normalized")
   
    return all_datasets, text_embedding_dim, image_embedding_dim
    

def train_mlp_aligner(text_model_name, image_model_name, device='cuda', 
                      num_epochs=None, batch_size=None, learning_rate=5e-3, 
                      save_dir='./aligner_checkpoints', architecture='single',
                      dataset_img_repr="imagenet1kval", few_shot_samples=None, 
                      filepath=None, pretrained_checkpoint_path=None,
                      **kwargs):

    # Get use_caption and other kwargs
    use_captions = kwargs.get('use_captions', False)
    no_weights = kwargs.get('no_weights', False)
    preprocess = kwargs.get('preprocess', None)
    loss_type = kwargs.get('loss_type', 'cosine')
    infonce_tau = kwargs.get('infonce_tau', 0.07)
    dropout_p = kwargs.get('dropout', 0.5)
    input_dropout_p = kwargs.get('input_dropout', 0.3)

    # Validate no_weights argument
    if no_weights and dataset_img_repr is None:
        raise ValueError("dataset_img_repr must not be None when no_weights=True")
    
    # Auto-select number of epochs if not specified
    if num_epochs is None:
        if image_model_name in IMAGENET1K_HEAD_MODELS:
            num_epochs = 2500
        else:
            num_epochs = 500
        print(f"Auto-selected {num_epochs} epochs for {image_model_name}")

    print(f"Training MLP aligner: {text_model_name} -> {image_model_name} (architecture: {architecture})")

    all_datasets, text_embedding_dim, image_embedding_dim = get_all_datasets(text_model_name, image_model_name, dataset_img_repr, few_shot_samples, no_weights, device, use_captions, preprocess=preprocess)
    
    dataloader = make_combined_loader(
        all_datasets, 
        batch_size=batch_size
    )
    
    print(f"✅ Combined dataloader created with global batch size: {batch_size}")
    print(f"Total samples in combined dataloader: {len(dataloader.dataset)}") 

    # Create MLP model
    model = TextToImageMLP(text_embedding_dim, image_embedding_dim, architecture=architecture, dropout=dropout_p, input_dropout_prob=input_dropout_p).to(device)
       
    # If a pretrained checkpoint is provided, load its weights
    if pretrained_checkpoint_path is not None:
        print(f"Loading pretrained model weights from: {pretrained_checkpoint_path}")
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained weights loaded. Continuing training from checkpoint.")


    # Loss function and optimizer with cosine annealing schedule
    # Loss function: CosineEmbeddingLoss or InfoNCE
    if loss_type == 'infonce':
        print(f"Using InfoNCE contrastive loss with tau={infonce_tau}")
    else:
        criterion = nn.CosineEmbeddingLoss()
        print("Using CosineEmbeddingLoss")
    
    # Use Adam for other architectures
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    print(f"Using Adam optimizer with lr={learning_rate}")

    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    model.train()
    # Ensure model is in float32
    model = model.float()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for text_emb, target_image_emb in dataloader:
            # Ensure both tensors are float32 and on correct device
            text_emb = text_emb.float().to(device)
            target_image_emb = target_image_emb.float().to(device)
            
            # Forward pass
            predicted_image_emb = model(text_emb)

            if loss_type == 'infonce':
                # Symmetric InfoNCE contrastive loss
                # Both embeddings are L2-normalized, matmul = cosine similarity
                sim = torch.matmul(predicted_image_emb, target_image_emb.T) / infonce_tau
                labels = torch.arange(sim.size(0), device=device)
                loss_i2t = F.cross_entropy(sim, labels)
                loss_t2i = F.cross_entropy(sim.T, labels)
                loss = (loss_i2t + loss_t2i) / 2
            else:
                # CosineEmbeddingLoss with target=+1 (embeddings should be similar)
                target = torch.ones(predicted_image_emb.size(0)).float().to(device)
                loss = criterion(predicted_image_emb, target_image_emb, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        if scheduler is not None:
            scheduler.step()  # Step the cosine annealing scheduler
        
        if (epoch + 1) % 100 == 0:  # Print every 100 epochs for longer training
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
    
    print(f"Training completed. Best loss: {best_loss:.6f} at epoch {best_epoch}")
    
    # Save the trained model
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'text_embedding_dim': text_embedding_dim,
            'image_embedding_dim': image_embedding_dim,
            'architecture': architecture
        },
        'training_config': {
            'text_model_name': text_model_name,
            'image_model_name': image_model_name,
            'dataset_img_repr': dataset_img_repr,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'few_shot_samples': few_shot_samples,
            'no_weights': no_weights
        },
        'timestamp': datetime.now().isoformat()
    }, filepath)
    
    print(f"Model saved to: {filepath}")
    
    return model, filepath

    
def train_cca_aligner(text_model_name, image_model_name, save_dir='./aligner_checkpoints', dataset_img_repr=None, 
                     few_shot_samples=None, filepath=None, batch_size = 64, **kwargs):
    """
    Train a CCA aligner and save it to the specified filepath.
    
    Args:
        text_model_name: Name of the text model
        image_model_name: Name of the image model 
        save_dir: Directory to save the CCA model (default: './aligner_checkpoints')
        dataset_img_repr: Dataset for image representations, if None only weights are used (default: None)
        few_shot_samples: Number of few-shot samples to use (default: None)
        no_weights: Whether to exclude weights embeddings (default: True)
        filepath: Full path where to save the CCA model (default: None)
        
    Returns:
        cca: Trained CCA object
    """
    from omegaconf import OmegaConf
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # No weights, use_captions, and preprocess from kwargs
    no_weights = kwargs.get('no_weights', False)
    use_captions = kwargs.get('use_captions', False)
    preprocess = kwargs.get('preprocess', None)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    all_datasets, text_embedding_dim, image_embedding_dim = get_all_datasets(text_model_name, image_model_name, dataset_img_repr, few_shot_samples, no_weights, device, use_captions, preprocess=preprocess)
    for idx, dataset in enumerate(all_datasets):
        # Check each dataset is non-empty
        if not dataset:
            print(f"Dataset {idx} is empty.")
            continue
        for j, subset in enumerate(dataset):
            if len(subset) > 0:
                first_element = subset[0]
                if isinstance(first_element, tuple) and j == 0:
                    print(f"  Subset {j}: First element shapes: {[elem.shape for elem in first_element]}")
                elif j==0:
                    print(f"  Subset {j}: First element shape: {first_element.shape}")
            else:
                print(f"  Subset {j}: Empty subset")

    concat_dataset = ConcatDataset(all_datasets)
    # Extract two tensors from the concatenated dataset, one for each modality
    train_emb_modal1 = []
    train_emb_modal2 = []

    for i in range(len(concat_dataset)):
        item = concat_dataset[i]
        if isinstance(item, tuple) and len(item) == 2:
            # Assert both modalities are non-empty
            assert item[0].numel() > 0, "Image embeddings are empty"
            assert item[1].numel() > 0, "Text embeddings are empty"
            # Let image be the first modality and text the second
            train_emb_modal1.append(item[1])
            train_emb_modal2.append(item[0])

    train_emb_modal1 = torch.stack(train_emb_modal1)
    train_emb_modal2 = torch.stack(train_emb_modal2)
    train_emb_modal1 = train_emb_modal1.cpu().numpy()
    train_emb_modal2 = train_emb_modal2.cpu().numpy()

    print(f"Training CCA with image embedding shape: {train_emb_modal1.shape}")
    print(f"Training CCA with text embedding shape: {train_emb_modal2.shape}")
    # Configure CCA
    cfg_dataset = OmegaConf.create({
        'sim_dim': 200,
        'equal_weights': False,
        'cca_proj_dims': [10, 50, 100, 200, 500, 750],
        'paths': {'save_path': './models'}
    })

    # Train CCA
    cca = NormalizedCCA()
    cca.fit_transform_train_data(cfg_dataset, train_emb_modal1, train_emb_modal2)

    # Save CCA model
    print(f'Saving CCA model to {filepath}')
    cca.save_model(filepath)
    
    return cca, filepath



def train_text2concepts_aligner(text_model_name, image_model_name, device='cuda', 
                      num_epochs=None, batch_size=None, learning_rate=5e-3, 
                      save_dir='./aligner_checkpoints', architecture=None,
                      dataset_img_repr="imagenet1kval", few_shot_samples=None,
                      filepath=None, **kwargs):
    
    # No weights, use_captions, and preprocess from kwargs
    no_weights = kwargs.get('no_weights', False)
    use_captions = kwargs.get('use_captions', False)
    preprocess = kwargs.get('preprocess', None)
    
    # Validate no_weights argument
    if no_weights and dataset_img_repr is None:
        raise ValueError("dataset_img_repr must not be None when no_weights=True")
    
    # Auto-select number of epochs if not specified
    if num_epochs is None:
        if image_model_name in IMAGENET1K_HEAD_MODELS:
            num_epochs = 2500
        else:
            num_epochs = 500
        print(f"Auto-selected {num_epochs} epochs for {image_model_name}")
    
    print(f"Training linear MLP aligner: {image_model_name} -> CLIP ViT-B/32")

    all_datasets, text_embedding_dim, image_embedding_dim = get_all_datasets(text_model_name, image_model_name, dataset_img_repr, few_shot_samples, no_weights, device, use_captions, preprocess=preprocess)

    dataloader = make_combined_loader(
        all_datasets, 
        batch_size=batch_size
    )
    
    print(f"✅ Combined dataloader created with global batch size: {batch_size}")
    print(f"Total samples in combined dataloader: {len(dataloader.dataset)}")    

    # Create MLP model (linear only)
    model = TextToImageMLP(text_embedding_dim, image_embedding_dim, architecture='single').to(device)
    print(f"Model created with image embedding dim: {image_embedding_dim}, text embedding dim: {text_embedding_dim}")
    # print keys of model
    print(f"Model keys: {list(model.state_dict().keys())}")
    # Loss function and optimizer with cosine annealing schedule
    criterion = nn.MSELoss()  # Use MSE loss for regression
    
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    print(f"Using Adam optimizer with lr={learning_rate}")
    
    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    model.train()
    # Ensure model is in float32
    model = model.float()
    best_loss = float('inf')

    print("Starting training...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for image_emb, target_clip_emb in dataloader:
            # Ensure both tensors are float32 and on correct device
            image_emb = image_emb.float().to(device)
            target_clip_emb = target_clip_emb.float().to(device)
            
            # Forward pass
            predicted_clip_emb = model(image_emb)
            loss = criterion(predicted_clip_emb, target_clip_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        if scheduler is not None:
            scheduler.step()  # Step the cosine annealing scheduler
        
        if (epoch + 1) % 100 == 0:  # Print every 100 epochs for longer training
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
        
    print(f"Training completed. Best loss: {best_loss:.6f} at epoch {best_epoch}")
    
    # Save the trained model
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'text_embedding_dim': text_embedding_dim,
            'image_embedding_dim': image_embedding_dim,
            'architecture': architecture
        },
        'training_config': {
            'text_model_name': text_model_name,
            'image_model_name': image_model_name,
            'dataset_img_repr': dataset_img_repr,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'adjusted_learning_rate': None,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'few_shot_samples': few_shot_samples,
            'no_weights': no_weights
        },
        'timestamp': datetime.now().isoformat()
    }, filepath)
    
    print(f"Model saved to: {filepath}")
    
    return model, filepath


def train_mlp_aligner_sequentially(text_model_name, image_model_name, device='cuda', 
                      num_epochs=None, batch_size=None, learning_rate=5e-4, 
                      save_dir='./aligner_checkpoints', architecture='two_layer',
                      dataset_img_repr="imagenet1kval", few_shot_samples=None, 
                      filepath=None, **kwargs):
    """
    Train MLP aligner using sequential two-stage training:
    Stage 1: Train on weights only (dataset_img_repr=None)
    Stage 2: Train on image representations only, using Stage 1 model as checkpoint
    """
    if dataset_img_repr is None:
        raise ValueError("dataset_img_repr must be provided for sequential training")
    
    # Stage 1: Check if weights-only model exists
    zs_aligner_filename = f"MLP_aligner_{architecture}_{image_model_name.replace('/', '_')}_{text_model_name}.pt"
    zs_aligner_filepath = os.path.join(save_dir, zs_aligner_filename)
    weights_exist = os.path.exists(zs_aligner_filepath)
    
    print("\n--- Stage 1: Training on weights only (dataset_img_repr=None) ---")
    if weights_exist:
        print(f"✅ Alignment weights already exist: {zs_aligner_filename}")
    else:
        # Train with dataset_img_repr=None (weights only)
        print("🔄 Training Stage 1: weights only...")
        _, zs_aligner_filepath = train_mlp_aligner(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            device=device,
            num_epochs=500,
            batch_size=batch_size,
            learning_rate=5e-3,
            save_dir=save_dir,
            architecture=architecture,
            dataset_img_repr=None,
            few_shot_samples=None,            
            filepath=zs_aligner_filepath,
            no_weights=False
        )
        print(f"✅ Stage 1 completed. Model saved to: {zs_aligner_filename}")
    
    # Stage 2: Check if sequential model exists
    stage2_filepath = get_mlp_aligner_path(
        dataset_img_repr=dataset_img_repr,
        few_shot_samples=few_shot_samples,
        architecture=architecture,
        image_model_name=image_model_name,
        text_model_name=text_model_name,
        save_dir=save_dir,
        no_weights=False,
        mode='MLP_sequential',
        preprocess=kwargs.get('preprocess', None)
    )

    assert stage2_filepath == filepath, f"Expected {filepath}, but got {stage2_filepath}"

    print("\n--- Stage 2: Training on image representations only (no_weights=True) ---")
    # Remove no_weights from kwargs if present to avoid duplicate keyword
    kwargs.pop('no_weights', None)
    final_model, final_model_path = train_mlp_aligner(
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=save_dir,
        architecture=architecture,
        dataset_img_repr=dataset_img_repr,
        few_shot_samples=few_shot_samples,
        pretrained_checkpoint_path=zs_aligner_filepath,
        filepath=stage2_filepath,
        no_weights=True,
        **kwargs  # Pass all kwargs including preprocess
    )
    print(f"✅ Stage 2 completed. Final model saved to: {final_model_path}")
        
    return final_model, final_model_path
