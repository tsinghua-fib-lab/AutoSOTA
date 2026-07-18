#!/usr/bin/env python3
"""
Gap Evaluation Script for Weight Representations

This script loads Timm models and extracts weight representations from their final layers
for ImageNet-1K classes, similar to the evaluation_wwobias_imagenet1k.py approach.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import argparse
import traceback
from datetime import datetime
from tqdm import tqdm

from config.config import IMAGENET21K_HEAD_MODELS, IMAGENET21K_HEAD_MODELS, GLOBAL_SEED, EMBEDDINGS_DIR, IMAGENET1K_HEAD_MODELS
from utils.utils import set_random_seeds, get_backbone, compute_and_save_mean_representation, apply_attention_based_preprocessing, apply_linear_projection_preprocessing
from utils.model_utils import (
    load_pretrained_model, _get_final_linear_layer, prepare_model_for_imagenet1k
)
from dataloaders.datasets_and_dataloaders import get_imagenet1kval_dataloaders
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats

class SimpleLinearClassifier(nn.Module):
    """
    Simple linear classifier for distinguishing between weight and image representations.
    """
    def __init__(self, input_dim, num_classes=2):
        super(SimpleLinearClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)

def load_weight_representations_imagenet1k(model_name, device='cuda', preprocess=None):
    """
    Load weight representations for ImageNet-1k classes from a pre-trained model.
    
    Args:
        model_name: Name of the model to load
        device: Device to use
        preprocess: Preprocessing mode ('mean', 'attention', 'linear', or None)
        
    Returns:
        Dictionary containing weight representations and metadata
    """
    print(f"\n{'='*80}")
    print(f"Loading weight representations for: {model_name}")
    if preprocess:
        print(f"Preprocessing mode: {preprocess}")
    print(f"{'='*80}")
    
    # Load complete pre-trained model
    print("Loading pre-trained model...")
    try:
        model = load_pretrained_model(model_name)
        print(f"Successfully loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None
    
    # Prepare model for ImageNet-1k (convert if needed)
    print("Preparing model for ImageNet-1k...")
    try:
        model, model_type, conversion_applied = prepare_model_for_imagenet1k(model, model_name, device)
        print(f"Model preparation complete: {model_type}")
    except Exception as e:
        print(f"Error preparing model {model_name}: {e}")
        print("This might be due to missing WNID mapping files or unsupported model architecture")
        return None
    
    model.to(device)
    
    # Extract weight representations from final linear layer
    print("Extracting weight representations...")
    try:
        last_linear = _get_final_linear_layer(model, model_name)
        
        # Get weights
        weight_vectors = last_linear.weight.detach()  # Keep on device for preprocessing
        
        # Determine dataset name for representations
        if model_name in IMAGENET21K_HEAD_MODELS:
            repr_dataset_name = "imagenet21k"
        else:
            repr_dataset_name = "imagenet1kval"

        # Apply preprocessing if specified
        if preprocess == 'mean':
            print("🔄 Applying mean-based preprocessing to weight vectors...")
            
            # Load the mean of representations
            mean_repr_name = f"imagenet1kval_reprmean_{model_name.replace('/', '_')}.pt"
            mean_repr_path = os.path.join(EMBEDDINGS_DIR, mean_repr_name)
            
            if not os.path.exists(mean_repr_path):
                print(f"⚠️  Mean representation file not found: {mean_repr_path}")
                print("🔄 Computing mean representation on-the-fly...")
                img_repr_mean, mean_repr_path = compute_and_save_mean_representation(model_name, device)
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
            from alignment.train_aligners import get_avg_image_embeddings
            
            # Get the name for the average representations for this image model
            img_repr_avg_name = f"imagenet1kval_reprmean_{model_name.replace('/', '_')}.pt"
            img_repr_avg_path = os.path.join(EMBEDDINGS_DIR, img_repr_avg_name)
            
            if not os.path.exists(img_repr_avg_path):
                print(f"⚠️  Average representation file not found: {img_repr_avg_path}")
                print(f"🔄 Computing imagenet1kval average representations on-the-fly...")
                # Determine num_classes based on dataset
                if repr_dataset_name == "imagenet21k":
                    num_classes_repr = 21841 # Approximate, get_avg_image_embeddings should handle it
                else:
                    num_classes_repr = 1000
                    
                img_reprs_avg, _ = get_avg_image_embeddings('imagenet1kval', model_name, num_classes=num_classes_repr, device=device, few_shot_samples=50, only_test=True)
            else:
                print(f"Loading imagenet1kval average representations from: {img_repr_avg_path}")
                img_reprs_avg = torch.load(img_repr_avg_path, map_location=device).float()

            # Apply attention-based preprocessing
            weight_vectors = apply_attention_based_preprocessing(weight_vectors, img_reprs_avg, device)
            # Note: The function already normalizes
            
        elif preprocess == 'linear':
            print("🔄 Applying linear projection preprocessing to weight vectors...")
            from alignment.train_aligners import get_avg_image_embeddings
            
            # Get the name for the average representations for this image model
            img_repr_avg_name = f"imagenet1kval_reprmean_{model_name.replace('/', '_')}.pt"
            img_repr_avg_path = os.path.join(EMBEDDINGS_DIR, img_repr_avg_name)
            
            if not os.path.exists(img_repr_avg_path):
                print(f"⚠️  Average representation file not found: {img_repr_avg_path}")
                print(f"🔄 Computing imagenet1kval average representations on-the-fly...")
                if repr_dataset_name == "imagenet21k":
                    num_classes_repr = 21841
                else:
                    num_classes_repr = 1000
                img_reprs_avg, _ = get_avg_image_embeddings('imagenet1kval', model_name, num_classes=num_classes_repr, device=device, few_shot_samples=None, only_test=True)
                torch.save(img_reprs_avg, img_repr_avg_path)
            else:
                print(f"Loading imagenet1kval average representations from: {img_repr_avg_path}")
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
            
        # Convert to numpy for the rest of the function
        weights = weight_vectors.cpu().numpy()  # Shape: [num_classes, feature_dim]
        
        print(f"Weight shape: {weights.shape}")
        
        # Detect zero rows (missing classes)
        zero_row_mask = np.all(weights == 0, axis=1)
        zero_row_indices = np.where(zero_row_mask)[0]
        non_zero_row_indices = np.where(~zero_row_mask)[0]
        
        print(f"Found {len(zero_row_indices)} zero rows (missing classes) out of {weights.shape[0]} total classes")
        if len(zero_row_indices) > 0:
            print(f"Zero row indices: {zero_row_indices[:10]}{'...' if len(zero_row_indices) > 10 else ''}")
        
        # Filter out zero rows
        filtered_weights = weights[non_zero_row_indices]
        
        print(f"Filtered weight shape: {filtered_weights.shape}")
        
        # Normalize weights across the feature dimension (L2 normalization)
        normalized_weights = filtered_weights / (np.linalg.norm(filtered_weights, axis=1, keepdims=True) + 1e-8)
        
        print(f"Normalized weight shape: {normalized_weights.shape}")
        
        # Create results dictionary (metadata only, weights saved separately)
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'conversion_applied': conversion_applied,
            'weight_shape': list(normalized_weights.shape),
            'num_classes': normalized_weights.shape[0],
            'feature_dim': normalized_weights.shape[1],
            'original_num_classes': weights.shape[0],
            'zero_row_indices': zero_row_indices.tolist(),
            'non_zero_row_indices': non_zero_row_indices.tolist(),
            'num_missing_classes': len(zero_row_indices)
        }
        
        # Add weight statistics (for normalized weights)
        results['weight_stats'] = {
            'min': float(normalized_weights.min()),
            'max': float(normalized_weights.max()),
            'mean': float(normalized_weights.mean()),
            'std': float(normalized_weights.std()),
            'l2_norms': np.linalg.norm(normalized_weights, axis=1, ord=2).tolist()  # L2 norm of each class weight
        }
        
        print(f"Normalized weight statistics:")
        print(f"  Min: {results['weight_stats']['min']:.4f}")
        print(f"  Max: {results['weight_stats']['max']:.4f}")
        print(f"  Mean: {results['weight_stats']['mean']:.4f}")
        print(f"  Std: {results['weight_stats']['std']:.4f}")
        
        # Store actual normalized weight data for saving
        results['_weight_data'] = normalized_weights
        
        return results
        
    except Exception as e:
        print(f"Error extracting weight representations: {e}")
        traceback.print_exc()
        return None
    finally:
        # Clean up
        del model
        # cleanup_model_cache()

def load_image_representations_imagenet1k(model_name, device='cuda', few_shot=5, batch_size=64):
    # ...existing code...
    print(f"\n{'='*80}")
    print(f"Loading image representations for: {model_name}")
    print(f"Few-shot: {few_shot} images per class")
    print(f"{'='*80}")
    
    # Create feature extractor using get_backbone function
    print("Creating feature extractor using get_backbone...")
    try:
        backbone, features_dim, preprocess = get_backbone(model_name)
        backbone.to(device)
        backbone.eval()
        print("Successfully created backbone feature extractor")
    except Exception as e:
        print(f"Error creating backbone feature extractor: {e}")
        return None
    
    # Get ImageNet-1K validation dataloader
    print("Loading ImageNet-1K validation dataset...")
    try:
        _, _, test_dataloader = get_imagenet1kval_dataloaders(batch_size, preprocess, only_test=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    
    # Collect image representations
    print("Extracting image features...")
    
    # Dictionary to store features per class
    class_features = {i: [] for i in range(1000)}  # ImageNet-1K has 1000 classes
    class_counts = {i: 0 for i in range(1000)}

    if few_shot is not None:
        total_images_needed = few_shot * 1000
    else:
        total_images_needed = None  # No limit
    images_collected = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_dataloader, desc="Extracting features")):
            if total_images_needed is not None and images_collected >= total_images_needed:
                break

            images = images.to(device)
            labels = labels.to(device)

            # Extract features using backbone
            features = backbone(images)
            # Normalize features
            features = F.normalize(features, p=2, dim=1)  # L2 normalization

            features = features.cpu().numpy()
            labels = labels.cpu().numpy()

            # Store features by class
            for i, (feature, label) in enumerate(zip(features, labels)):
                if few_shot is None or class_counts[label] < few_shot:
                    class_features[label].append(feature)
                    class_counts[label] += 1
                    images_collected += 1

            # Check if we have enough images for all classes (only if few_shot is set)
            if few_shot is not None and all(count >= few_shot for count in class_counts.values()):
                break
    
    # Convert to numpy arrays and compute averaged class representations
    print("Computing averaged class representations...")
    
    averaged_class_features = {}  # Will store averaged representations per class
    all_averaged_features = []
    feature_dim = None
    valid_classes = 0
    
    for class_idx in range(1000):
        if len(class_features[class_idx]) > 0:
            class_array = np.array(class_features[class_idx])
            if feature_dim is None:
                feature_dim = class_array.shape[1]
            
            # Compute average representation for this class
            class_avg = class_array.mean(axis=0)
            averaged_class_features[class_idx] = class_avg
            all_averaged_features.append(class_avg)
            valid_classes += 1
    
    all_averaged_features = np.array(all_averaged_features) if all_averaged_features else np.array([])
    
    # Create results dictionary (metadata only, features saved separately)
    results = {
        'model_name': model_name,
        'few_shot': few_shot,
        'feature_dim': feature_dim,
        'total_classes': valid_classes,  # Number of classes with averaged representations
        'total_original_images': sum(len(class_features[i]) for i in range(1000)),  # Total individual images used
        'images_per_class': {str(i): len(class_features[i]) for i in range(1000) if len(class_features[i]) > 0}
    }
    
    # Add feature statistics for averaged representations
    if len(all_averaged_features) > 0:
        results['feature_stats'] = {
            'min': float(all_averaged_features.min()),
            'max': float(all_averaged_features.max()),
            'mean': float(all_averaged_features.mean()),
            'std': float(all_averaged_features.std())
        }
        
        # Per-class statistics (means and stds of original samples before averaging)
        class_means = []
        class_stds = []
        for class_idx in range(1000):
            if len(class_features[class_idx]) > 0:
                class_array = np.array(class_features[class_idx])
                class_means.append(class_array.mean(axis=0))
                class_stds.append(class_array.std(axis=0))
        
        if class_means:
            results['class_feature_stats'] = {
                'class_means': [mean.tolist() for mean in class_means],
                'class_stds': [std.tolist() for std in class_stds]
            }
    
    print(f"Image feature extraction complete:")
    print(f"  Total classes with averaged representations: {valid_classes}")
    print(f"  Total original images used: {results['total_original_images']}")
    print(f"  Feature dimension: {feature_dim}")
    if few_shot is not None:
        print(f"  Classes with {few_shot} images: {sum(1 for count in class_counts.values() if count >= few_shot)}/1000")
    
    # Store averaged feature data for saving
    results['_class_features'] = averaged_class_features

    del backbone

    return results

def filter_image_representations_by_valid_classes(image_results, weight_results):
    if weight_results is None or 'non_zero_row_indices' not in weight_results:
        print("No weight filtering information available - returning original image results")
        return image_results
    
    valid_class_indices = set(weight_results['non_zero_row_indices'])
    print(f"Filtering image representations to {len(valid_class_indices)} valid classes...")
    
    # Filter class features from the actual data (now contains averaged representations)
    original_class_features = image_results['_class_features']
    filtered_class_features = {}
    filtered_images_per_class = {}
    total_filtered_classes = 0
    
    for class_idx in range(1000):
        if class_idx in valid_class_indices and class_idx in original_class_features:
            class_idx_str = str(class_idx)
            filtered_class_features[class_idx] = original_class_features[class_idx]
            # Keep track of how many original images were used for this averaged representation
            if class_idx_str in image_results['images_per_class']:
                filtered_images_per_class[class_idx_str] = image_results['images_per_class'][class_idx_str]
            total_filtered_classes += 1
    
    # Create filtered results
    filtered_results = image_results.copy()
    filtered_results['images_per_class'] = filtered_images_per_class
    filtered_results['total_classes'] = total_filtered_classes  # Number of valid averaged representations
    filtered_results['num_valid_classes'] = len(filtered_class_features)
    filtered_results['filtered_class_indices'] = list(valid_class_indices)
    filtered_results['original_total_classes'] = image_results.get('total_classes', 0)
    
    # Store filtered feature data
    filtered_results['_class_features'] = filtered_class_features
    
    # Recompute statistics for filtered averaged representations
    all_filtered_features = []
    for features in filtered_class_features.values():
        all_filtered_features.append(features)  # Each is already an averaged vector
    
    if all_filtered_features:
        all_filtered_features = np.array(all_filtered_features)
        filtered_results['feature_stats'] = {
            'min': float(all_filtered_features.min()),
            'max': float(all_filtered_features.max()),
            'mean': float(all_filtered_features.mean()),
            'std': float(all_filtered_features.std())
        }
        
        # Store the averaged representations as the class means 
        # (Note: no per-class std since we now have one averaged vector per class)
        class_averaged_features = []
        for features in filtered_class_features.values():
            class_averaged_features.append(features)  # Each is already an averaged vector
        
        if class_averaged_features:
            filtered_results['class_feature_stats'] = {
                'class_averaged_features': [feat.tolist() for feat in class_averaged_features]
            }
    print(f"Image filtering complete:")
    print(f"  Original classes: {len(image_results['_class_features'])}")
    print(f"  Filtered classes: {len(filtered_class_features)}")
    print(f"  Original total images used: {image_results.get('total_original_images', 'N/A')}")
    print(f"  Filtered classes with averaged representations: {total_filtered_classes}")
    
    return filtered_results

def save_representations_to_files(model_results, model_name, output_dir):
    # Create safe filename
    safe_model_name = model_name.replace('/', '_').replace(':', '_')
    
    result_files = {}
    cleaned_results = {}
    
    # Save weights
    if 'weights' in model_results and '_weight_data' in model_results['weights']:
        weight_file = os.path.join(output_dir, f"{safe_model_name}_weights.pt")
        weight_data = {
            'weights': model_results['weights']['_weight_data'],
        }
        torch.save(weight_data, weight_file)
        result_files['weights_file'] = weight_file
        
        # Clean metadata (remove actual data)
        cleaned_weights = model_results['weights'].copy()
        del cleaned_weights['_weight_data']
        cleaned_results['weights'] = cleaned_weights
        print(f"Saved weights to: {weight_file}")
    
    # Save images
    if 'images' in model_results and '_class_features' in model_results['images']:
        images_file = os.path.join(output_dir, f"{safe_model_name}_images.pt")
        image_data = {
            'class_features': model_results['images']['_class_features']
        }
        torch.save(image_data, images_file)
        result_files['images_file'] = images_file
        
        # Clean metadata (remove actual data)
        cleaned_images = model_results['images'].copy()
        del cleaned_images['_class_features']
        cleaned_results['images'] = cleaned_images
        print(f"Saved images to: {images_file}")
    
    # Save unfiltered images if they exist
    if 'images_unfiltered' in model_results and '_class_features' in model_results['images_unfiltered']:
        images_unfiltered_file = os.path.join(output_dir, f"{safe_model_name}_images_unfiltered.pt")
        image_unfiltered_data = {
            'class_features': model_results['images_unfiltered']['_class_features']
        }
        torch.save(image_unfiltered_data, images_unfiltered_file)
        result_files['images_unfiltered_file'] = images_unfiltered_file
        
        # Clean metadata
        cleaned_images_unfiltered = model_results['images_unfiltered'].copy()
        del cleaned_images_unfiltered['_class_features']
        cleaned_results['images_unfiltered'] = cleaned_images_unfiltered
        print(f"Saved unfiltered images to: {images_unfiltered_file}")
    
    # Add file paths to cleaned results
    cleaned_results['data_files'] = result_files
    
    return cleaned_results

def load_representations_from_files(cleaned_results):
    loaded_results = cleaned_results.copy()
    
    if 'data_files' in cleaned_results:
        data_files = cleaned_results['data_files']
        
        # Load weights
        if 'weights_file' in data_files and os.path.exists(data_files['weights_file']):
            weight_data = torch.load(data_files['weights_file'], map_location='cpu')
            # Normalize weights using L2 norm
            if 'weights' in loaded_results:
                loaded_results['weights']['_weight_data'] = weight_data['weights'] / (np.linalg.norm(weight_data['weights'], axis=1, keepdims=True) + 1e-8)
        
        # Load images
        if 'images_file' in data_files and os.path.exists(data_files['images_file']):
            image_data = torch.load(data_files['images_file'], map_location='cpu')
            # Normalize each image vector using L2 norm
            if 'images' in loaded_results:
                normalized_class_features = {}
                for k, v in image_data['class_features'].items():
                    arr = np.array(v)
                    norm = np.linalg.norm(arr) + 1e-8
                    normalized_class_features[k] = arr / norm
                loaded_results['images']['_class_features'] = normalized_class_features

        # Load unfiltered images
        if 'images_unfiltered_file' in data_files and os.path.exists(data_files['images_unfiltered_file']):
            image_unfiltered_data = torch.load(data_files['images_unfiltered_file'], map_location='cpu')
            if 'images_unfiltered' in loaded_results:
                loaded_results['images_unfiltered']['_class_features'] = image_unfiltered_data['class_features']
    
    return loaded_results

def extract_multiple_model_representations(model_names, device='cuda', output_dir='./weight_representations', 
                                         extract_weights=True, extract_images=False, few_shot=5, batch_size=64):
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for i, model_name in enumerate(model_names):
        print(f"\n\nProcessing model {i+1}/{len(model_names)}: {model_name}")
        
        model_results = {}
        weight_results = None
        
        try:
            if extract_weights:
                print("Extracting weight representations...")
                weight_results = load_weight_representations_imagenet1k(model_name, device)
                if weight_results:
                    model_results['weights'] = weight_results
                    print(f"✅ Weight extraction successful for {model_name}")
                else:
                    print(f"❌ Weight extraction failed for {model_name}")
            
            if extract_images:
                print("Extracting image representations...")
                image_results = load_image_representations_imagenet1k(model_name, device, few_shot, batch_size)
                print(f"Image extraction results keys: {image_results.keys()}")
                if image_results:
                    # Filter image representations based on valid weight classes
                    if weight_results is not None:
                        print("Filtering image representations to match valid weight classes...")
                        filtered_image_results = filter_image_representations_by_valid_classes(image_results, weight_results)
                        model_results['images'] = filtered_image_results
                        model_results['images_unfiltered'] = image_results  # Keep original for reference
                    else:
                        model_results['images'] = image_results
                    print(f"✅ Image extraction successful for {model_name}")
                else:
                    print(f"❌ Image extraction failed for {model_name}")
            
            if model_results:
                # Save representations to .pt files and get cleaned metadata
                cleaned_results = save_representations_to_files(model_results, model_name, output_dir)
                all_results[model_name] = cleaned_results
                print(f"✅ Successfully processed {model_name}")
            else:
                print(f"❌ Failed to process {model_name}")
                
        except Exception as e:
            print(f"❌ Error processing {model_name}: {str(e)}")
            traceback.print_exc()
            all_results[model_name] = {'error': str(e)}
        
        # Save intermediate metadata results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = os.path.join(output_dir, f"metadata_intermediate_{timestamp}.json")
        with open(intermediate_file, 'w') as f:
            json.dump(make_json_serializable(all_results), f, indent=2)
    
    return all_results

def train_weight_image_classifier(weight_data, image_data, device='cuda', 
                                train_split=0.8, epochs=100, batch_size=64, lr=0.001):
    print(f"\n{'='*60}")
    print("Training Linear Classifier: Weight vs Averaged Image Representations")
    print(f"{'='*60}")
    
    # Prepare data
    print("Preparing data...")
    print(f"Weight data shape: {weight_data.shape} (one vector per class)")
    print(f"Image data shape: {image_data.shape} (one averaged vector per class)")

    # Check if dimensions match
    if weight_data.shape[1] != image_data.shape[1]:
        print(f"ERROR: Feature dimensions don't match!")
        print(f"Weight features: {weight_data.shape[1]}, Image features: {image_data.shape[1]}")
        return None

    feature_dim = weight_data.shape[1]

    # Create labels: 0 for weights, 1 for images
    weight_labels = np.zeros(weight_data.shape[0])
    image_labels = np.ones(image_data.shape[0])

    # Combine data
    X = np.vstack([weight_data, image_data])
    y = np.hstack([weight_labels, image_labels])

    print(f"Total samples: {X.shape[0]} (weight classes: {len(weight_labels)}, image classes: {len(image_labels)})")
    print(f"Feature dimension: {feature_dim}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_split, random_state=42, stratify=y
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # --- Standard classification ---
    model = SimpleLinearClassifier(input_dim=feature_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nTraining for {epochs} epochs...")
    model.train()
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_losses = []
    train_accuracies = []
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")
        for batch_X, batch_y in batch_pbar:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            batch_acc = 100.0 * correct / total
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.2f}'})
        epoch_loss /= len(train_loader)
        epoch_acc = 100.0 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        epoch_pbar.set_postfix({'loss': f'{epoch_loss:.4f}', 'acc': f'{epoch_acc:.2f}'})
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Evaluation
    print(f"\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = 100.0 * (test_predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        test_probs = F.softmax(test_outputs, dim=1)
        y_test_np = y_test_tensor.cpu().numpy()
        test_predicted_np = test_predicted.cpu().numpy()
        test_probs_np = test_probs.cpu().numpy()

    print(f"\n{'='*60}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"\nDetailed Classification Report:")
    class_names = ['Weight Representations', 'Averaged Image Representations']
    print(classification_report(y_test_np, test_predicted_np, target_names=class_names))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test_np, test_predicted_np)
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                Weight  Image")
    print(f"Actual Weight   {cm[0,0]:6d}  {cm[0,1]:5d}")
    print(f"       Image    {cm[1,0]:6d}  {cm[1,1]:5d}")
    weight_indices = y_test_np == 0
    image_indices = y_test_np == 1
    weight_confidence = test_probs_np[weight_indices, 0].mean()
    image_confidence = test_probs_np[image_indices, 1].mean()
    print(f"\nPrediction Confidence:")
    print(f"Average confidence for weight predictions: {weight_confidence:.3f}")
    print(f"Average confidence for image predictions: {image_confidence:.3f}")

    # --- Randomized-label classification ---
    print(f"\n{'='*60}")
    print("RANDOMIZED LABEL CLASSIFICATION")
    print(f"{'='*60}")
    # Randomly shuffle training labels with probability 0.5
    rng = np.random.default_rng(GLOBAL_SEED)
    y_train_random = y_train.copy()
    mask = rng.random(y_train_random.shape[0]) < 0.5
    y_train_random[mask] = 1 - y_train_random[mask]

    # Train new model on randomized labels
    model_rand = SimpleLinearClassifier(input_dim=feature_dim).to(device)
    optimizer_rand = torch.optim.Adam(model_rand.parameters(), lr=lr)
    train_dataset_rand = torch.utils.data.TensorDataset(X_train_tensor, torch.LongTensor(y_train_random).to(device))
    train_loader_rand = torch.utils.data.DataLoader(train_dataset_rand, batch_size=batch_size, shuffle=True)
    train_losses_rand = []
    train_accuracies_rand = []
    epoch_pbar_rand = tqdm(range(epochs), desc="RandLabel Training", unit="epoch")
    for epoch in epoch_pbar_rand:
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_pbar = tqdm(train_loader_rand, desc=f"RandLabel Epoch {epoch+1}", leave=False, unit="batch")
        for batch_X, batch_y in batch_pbar:
            optimizer_rand.zero_grad()
            outputs = model_rand(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer_rand.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            batch_acc = 100.0 * correct / total
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.2f}'})
        epoch_loss /= len(train_loader_rand)
        epoch_acc = 100.0 * correct / total
        train_losses_rand.append(epoch_loss)
        train_accuracies_rand.append(epoch_acc)
        epoch_pbar_rand.set_postfix({'loss': f'{epoch_loss:.4f}', 'acc': f'{epoch_acc:.2f}'})
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"RandLabel Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Evaluation on original test set
    model_rand.eval()
    with torch.no_grad():
        test_outputs_rand = model_rand(X_test_tensor)
        test_loss_rand = criterion(test_outputs_rand, y_test_tensor).item()
        _, test_predicted_rand = torch.max(test_outputs_rand.data, 1)
        test_accuracy_rand = 100.0 * (test_predicted_rand == y_test_tensor).sum().item() / y_test_tensor.size(0)
        test_probs_rand = F.softmax(test_outputs_rand, dim=1)
        test_probs_rand_np = test_probs_rand.cpu().numpy()
    print(f"RandLabel Test Loss: {test_loss_rand:.4f}")
    print(f"RandLabel Test Accuracy: {test_accuracy_rand:.2f}%")
    print(f"\nDetailed Classification Report (RandLabel):")
    print(classification_report(y_test_np, test_predicted_rand.cpu().numpy(), target_names=class_names))
    cm_rand = confusion_matrix(y_test_np, test_predicted_rand.cpu().numpy())
    print(f"\nConfusion Matrix (RandLabel):")
    print(f"                    Predicted")
    print(f"                Weight  Image")
    print(f"Actual Weight   {cm_rand[0,0]:6d}  {cm_rand[0,1]:5d}")
    print(f"       Image    {cm_rand[1,0]:6d}  {cm_rand[1,1]:5d}")
    weight_confidence_rand = test_probs_rand_np[weight_indices, 0].mean()
    image_confidence_rand = test_probs_rand_np[image_indices, 1].mean()
    print(f"\nPrediction Confidence (RandLabel):")
    print(f"Average confidence for weight predictions: {weight_confidence_rand:.3f}")
    print(f"Average confidence for image predictions: {image_confidence_rand:.3f}")

    # Return results
    results = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'final_train_accuracy': train_accuracies[-1],
        'final_train_loss': train_losses[-1],
        'confusion_matrix': cm.tolist(),
        'weight_confidence': float(weight_confidence),
        'image_confidence': float(image_confidence),
        'num_train_samples': X_train.shape[0],
        'num_test_samples': X_test.shape[0],
        'feature_dim': feature_dim,
        'epochs_trained': epochs,
        'randlabel': {
            'test_accuracy': test_accuracy_rand,
            'test_loss': test_loss_rand,
            'final_train_accuracy': train_accuracies_rand[-1],
            'final_train_loss': train_losses_rand[-1],
            'confusion_matrix': cm_rand.tolist(),
            'weight_confidence': float(weight_confidence_rand),
            'image_confidence': float(image_confidence_rand)
        }
    }
    return results
    
    weight_confidence = test_probs_np[weight_indices, 0].mean()
    image_confidence = test_probs_np[image_indices, 1].mean()
    
    print(f"\nPrediction Confidence:")
    print(f"Average confidence for weight predictions: {weight_confidence:.3f}")
    print(f"Average confidence for image predictions: {image_confidence:.3f}")
    
    # Return results
    results = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'final_train_accuracy': train_accuracies[-1],
        'final_train_loss': train_losses[-1],
        'confusion_matrix': cm.tolist(),
        'weight_confidence': float(weight_confidence),
        'image_confidence': float(image_confidence),
        'num_train_samples': X_train.shape[0],
        'num_test_samples': X_test.shape[0],
        'feature_dim': feature_dim,
        'epochs_trained': epochs
    }
    
    return results

def run_classification_analysis(all_results, device='cuda', output_dir='./representations'):
    print(f"\n{'='*80}")
    print("RUNNING CLASSIFICATION ANALYSIS")
    print(f"{'='*80}")
    
    classification_results = {}
    
    for model_name, model_results in all_results.items():
        if 'error' in model_results:
            print(f"\nSkipping {model_name} due to error")
            continue
        
        if 'weights' not in model_results or 'images' not in model_results:
            print(f"\nSkipping {model_name} - missing weight or image data")
            continue
        
        print(f"\n{'-'*60}")
        print(f"Analyzing model: {model_name}")
        print(f"{'-'*60}")
        
        try:
            # Load representations from files
            loaded_results = load_representations_from_files(model_results)
            
            # Extract weight data
            if '_weight_data' not in loaded_results['weights']:
                print(f"Could not load weight data for {model_name}")
                continue
            weight_data = loaded_results['weights']['_weight_data']
            print(f"Weight data shape: {weight_data.shape}")
            
            # Extract image data - collect averaged representations per class
            if '_class_features' not in loaded_results['images']:
                print(f"Could not load image data for {model_name}")
                continue
            
            image_features_list = []
            for class_idx, class_avg_features in loaded_results['images']['_class_features'].items():
                image_features_list.append(class_avg_features)  # Each is already an averaged vector
            
            image_data = np.array(image_features_list)
            print(f"Image data shape: {image_data.shape}")
            print(f"Image data represents {len(image_features_list)} class-averaged representations")
            
            # Check if we have enough data
            if weight_data.shape[0] < 10 or image_data.shape[0] < 10:
                print(f"Insufficient data for classification (weights: {weight_data.shape[0]}, images: {image_data.shape[0]})")
                continue
            
            # Train classifier
            results = train_weight_image_classifier(
                weight_data, image_data, device=device,
                train_split=0.8, epochs=100, batch_size=64, lr=0.001
            )
            
            if results:
                classification_results[model_name] = results
                print(f"✅ Classification completed for {model_name}")
                
                # Note: Permutation test is now handled separately via run_permutation_test_analysis
                print("Permutation test is handled separately via --permutation_test flag")
                
             
            else:
                print(f"❌ Classification failed for {model_name}")
                
        except Exception as e:
            print(f"❌ Error in classification for {model_name}: {e}")
            traceback.print_exc()
    
    return classification_results

def run_permutation_test_analysis(all_results, device='cuda', output_dir='./representations', 
                                  n_permutations=10000, permutation_seed=42):
    print(f"\n{'='*80}")
    print("RUNNING PERMUTATION TEST ANALYSIS")
    print(f"{'='*80}")
    
    permutation_results = {}
    
    for model_name, model_results in all_results.items():
        if 'error' in model_results:
            print(f"\nSkipping {model_name} due to error")
            continue
        
        if 'weights' not in model_results or 'images' not in model_results:
            print(f"\nSkipping {model_name} - missing weight or image data")
            continue
        
        print(f"\n{'-'*60}")
        print(f"Running permutation test for: {model_name}")
        print(f"{'-'*60}")
        
        try:
            # Load representations from files
            loaded_results = load_representations_from_files(model_results)
            
            # Extract weight data
            if '_weight_data' not in loaded_results['weights']:
                print(f"Could not load weight data for {model_name}")
                continue
            weight_data = loaded_results['weights']['_weight_data']
            print(f"Weight data shape: {weight_data.shape}")
            
            # Extract image data - collect averaged representations per class
            if '_class_features' not in loaded_results['images']:
                print(f"Could not load image data for {model_name}")
                continue
            
            image_features_list = []
            for class_idx, class_avg_features in loaded_results['images']['_class_features'].items():
                image_features_list.append(class_avg_features)
            
            image_data = np.array(image_features_list)
            print(f"Image data shape: {image_data.shape}")
            
            # Check if we have enough data
            if weight_data.shape[0] < 10 or image_data.shape[0] < 10:
                print(f"Insufficient data for permutation test (weights: {weight_data.shape[0]}, images: {image_data.shape[0]})")
                continue
            
            # Run permutation test
            print("Computing centroids and running permutation test...")
            try:
                centroid_results = compute_centroids_and_permutation_test(
                    weight_data, image_data, model_name, 
                    n_permutations=n_permutations, random_seed=permutation_seed
                )
                if centroid_results:
                    permutation_results[model_name] = centroid_results
                    print(f"✅ Permutation test completed for {model_name}")
                                
                else:
                    print(f"⚠️ Permutation test failed for {model_name}")
            except Exception as e:
                print(f"⚠️ Error in permutation test for {model_name}: {e}")
                
        except Exception as e:
            print(f"❌ Error in permutation test analysis for {model_name}: {e}")
            traceback.print_exc()
    
    return permutation_results

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def compute_centroids_and_permutation_test(weight_data, image_data, model_name, n_permutations=10000, random_seed=42):
    print(f"\n{'='*60}")
    print(f"Computing Centroids and Permutation Test for: {model_name}")
    print(f"{'='*60}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Ensure both datasets have the same number of classes
    n = weight_data.shape[0]
    m = image_data.shape[0]
    if n == 0 or m == 0:
        print("No data for permutation test.")
        return None
    pooled = np.vstack([weight_data, image_data])
    labels = np.array([0]*n + [1]*m)

    # Observed distance
    weight_centroid = np.mean(weight_data, axis=0)
    image_centroid = np.mean(image_data, axis=0)
    observed_distance = np.linalg.norm(weight_centroid - image_centroid)

    print(f"Weight centroid norm: {np.linalg.norm(weight_centroid):.4f}")
    print(f"Image centroid norm: {np.linalg.norm(image_centroid):.4f}")
    print(f"Observed distance between centroids: {observed_distance:.4f}")

    # Permutation test
    print(f"Running permutation test with {n_permutations} permutations...")
    rng = np.random.default_rng(random_seed)
    permuted_distances = np.empty(n_permutations)
    for k in tqdm(range(n_permutations), desc="Permutation test"):
        rng.shuffle(labels)
        grp1 = pooled[labels == 0]
        grp2 = pooled[labels == 1]
        permuted_distances[k] = np.linalg.norm(np.mean(grp1, axis=0) - np.mean(grp2, axis=0))

    # Calculate p-value (one-tailed test: how often do random permutations give distances >= observed)
    p_value = (np.sum(permuted_distances >= observed_distance) + 1) / (n_permutations + 1)

    # Additional statistics
    mean_permuted_distance = np.mean(permuted_distances)
    std_permuted_distance = np.std(permuted_distances)
    min_permuted_distance = np.min(permuted_distances)
    max_permuted_distance = np.max(permuted_distances)

    # Effect size (Cohen's d-like measure)
    if std_permuted_distance > 0:
        effect_size = (observed_distance - mean_permuted_distance) / std_permuted_distance
    else:
        effect_size = float('inf') if observed_distance != mean_permuted_distance else 0.0

    # Percentile of observed distance
    percentile = (np.sum(permuted_distances < observed_distance) / n_permutations) * 100

    # Debug information
    print(f"\nDEBUG INFORMATION:")
    print(f"Permuted distances range: [{min_permuted_distance:.4f}, {max_permuted_distance:.4f}]")
    print(f"Permuted distances std: {std_permuted_distance:.4f}")
    print(f"Number of permuted distances >= observed: {np.sum(permuted_distances >= observed_distance)}")
    print(f"Number of permuted distances < observed: {np.sum(permuted_distances < observed_distance)}")

    print(f"\n{'='*60}")
    print("PERMUTATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Observed distance: {observed_distance:.4f}")
    print(f"Mean permuted distance: {mean_permuted_distance:.4f}")
    print(f"Std permuted distance: {std_permuted_distance:.4f}")
    print(f"Min permuted distance: {min_permuted_distance:.4f}")
    print(f"Max permuted distance: {max_permuted_distance:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Effect size: {effect_size:.4f}")
    print(f"Observed distance percentile: {percentile:.2f}%")

    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"

    print(f"Significance: {significance}")

    # Create results dictionary
    results = {
        'model_name': model_name,
        'num_classes': n + m,
        'feature_dim': weight_data.shape[1],
        'weight_centroid': weight_centroid.tolist(),
        'image_centroid': image_centroid.tolist(),
        'weight_centroid_norm': float(np.linalg.norm(weight_centroid)),
        'image_centroid_norm': float(np.linalg.norm(image_centroid)),
        'observed_distance': float(observed_distance),
        'permutation_test': {
            'n_permutations': n_permutations,
            'p_value': float(p_value),
            'mean_permuted_distance': float(mean_permuted_distance),
            'std_permuted_distance': float(std_permuted_distance),
            'min_permuted_distance': float(min_permuted_distance),
            'max_permuted_distance': float(max_permuted_distance),
            'effect_size': float(effect_size),
            'percentile': float(percentile),
            'significance': significance,
            'num_permuted_greater_equal': int(np.sum(permuted_distances >= observed_distance)),
            'num_permuted_less': int(np.sum(permuted_distances < observed_distance))
        },
        'permuted_distances_sample': permuted_distances[:1000].tolist()  # Store first 1000 for plotting
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract and analyze weight and image representations from models")
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='List of model names to process (space-separated)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (default: cuda)')
    parser.add_argument('--output_dir', type=str, default='./weight_representations',
                        help='Directory to save results (default: ./weight_representations)')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED,
                        help='Random seed (default: global seed)')
    parser.add_argument('--extract_weights', action='store_true',
                        help='Extract weight representations')
    parser.add_argument('--extract_images', action='store_true',
                        help='Extract image representations')
    parser.add_argument('--few_shot', type=int, default=None,
                        help='Number of images per class for image representations (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for image processing (default: 64)')
    parser.add_argument('--skip_classification', action='store_true',
                        help='Skip the classification analysis even if both weights and images are extracted')
    parser.add_argument('--classification', action='store_true',
                        help='Run classification analysis between weight and image representations')
    parser.add_argument('--permutation_test', action='store_true',
                        help='Run permutation test for centroid analysis')
    parser.add_argument('--n_permutations', type=int, default=10000,
                        help='Number of permutations for the permutation test (default: 10000)')
    parser.add_argument('--permutation_seed', type=int, default=42,
                        help='Random seed for permutation test (default: 42)')
    
    args = parser.parse_args()
    
    # Set default behavior: if neither flag is specified, extract both
    if not args.extract_weights and not args.extract_images:
        args.extract_weights = True
        args.extract_images = True
        print("No extraction flags specified - defaulting to extract both weights and images")
    
    # Set random seed
    set_random_seeds(args.seed)
    
    # Determine which models to process
    if args.models:
        models_to_process = args.models
        print(f"Processing specified models: {models_to_process}")
    else:
        models_to_process = [IMAGENET21K_HEAD_MODELS[0]]
        print(f"Processing all IMAGENET21K_HEAD_MODELS: {len(models_to_process)} models")
    
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Extract weights: {args.extract_weights}")
    print(f"Extract images: {args.extract_images}")
    print(f"Run classification: {args.classification}")
    print(f"Run permutation test: {args.permutation_test}")
    if args.extract_images:
        print(f"Few-shot: {args.few_shot}")
        print(f"Batch size: {args.batch_size}")
    
    # Check if classification will run
    will_run_classification = args.classification and args.extract_weights and args.extract_images and not args.skip_classification
    print(f"Will run classification analysis: {will_run_classification}")
    if args.classification and not (args.extract_weights and args.extract_images):
        print("WARNING: Classification requires both --extract_weights and --extract_images")

    # Extract representations
    all_results = extract_multiple_model_representations(
        models_to_process, args.device, args.output_dir, 
        args.extract_weights, args.extract_images, args.few_shot, args.batch_size
    )

    # Run classification analysis if both weights and images are extracted
    classification_results = {}
    if will_run_classification:
        print(f"\n{'='*80}")
        print("STARTING CLASSIFICATION ANALYSIS")
        print(f"{'='*80}")
        classification_results = run_classification_analysis(all_results, args.device, args.output_dir)
        
        # Save classification results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        classification_file = os.path.join(args.output_dir, f"classification_results_{timestamp}.json")
        with open(classification_file, 'w') as f:
            json.dump(make_json_serializable(classification_results), f, indent=2)
        print(f"Classification results saved to: {classification_file}")
    else:
        if args.skip_classification:
            print("Classification analysis skipped due to --skip_classification flag")
        elif not args.classification:
            print("Classification analysis skipped - use --classification to enable")
        elif not args.extract_weights:
            print("Classification analysis skipped - weights not extracted")
        elif not args.extract_images:
            print("Classification analysis skipped - images not extracted")
    
    # Run permutation test analysis if requested (independent of classification)
    permutation_results = {}
    will_run_permutation_test = args.permutation_test and args.extract_weights and args.extract_images
    print(f"Will run permutation test analysis: {will_run_permutation_test}")
    if will_run_permutation_test:
        print(f"\n{'='*80}")
        print("STARTING PERMUTATION TEST ANALYSIS")
        print(f"{'='*80}")
        permutation_results = run_permutation_test_analysis(all_results, args.device, args.output_dir, 
                                                           args.n_permutations, args.permutation_seed)
        
        # Save permutation test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        permutation_file = os.path.join(args.output_dir, f"permutation_results_{timestamp}.json")
        with open(permutation_file, 'w') as f:
            json.dump(make_json_serializable(permutation_results), f, indent=2)
        print(f"Permutation test results saved to: {permutation_file}")
    else:
        if not args.permutation_test:
            print("Permutation test analysis skipped - use --permutation_test to enable")
        elif not args.extract_weights:
            print("Permutation test analysis skipped - weights not extracted")
        elif not args.extract_images:
            print("Permutation test analysis skipped - images not extracted")
    
    # Save final metadata results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = os.path.join(args.output_dir, f"metadata_{timestamp}.json")
    
    with open(metadata_file, 'w') as f:
        json.dump(make_json_serializable(all_results), f, indent=2)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Representation data saved as .pt files in: {args.output_dir}")
    
    # Print summary
    successful_models = [m for m, r in all_results.items() if 'error' not in r]
    failed_models = [m for m, r in all_results.items() if 'error' in r]
    
    print(f"\nSummary:")
    print(f"  Total models: {len(models_to_process)}")
    print(f"  Successful: {len(successful_models)}")
    print(f"  Failed: {len(failed_models)}")
    
    if failed_models:
        print(f"\nFailed models:")
        for model in failed_models:
            print(f"  - {model}")
    
    if successful_models:
        if args.extract_weights:
            print(f"\nWeight Representation Summary:")
            print(f"{'Model':<50} {'Classes':<8} {'Missing':<8} {'Features':<10} {'Weight Range':<15}")
            print("-" * 95)
            
            for model in successful_models:
                if 'weights' in all_results[model] and 'weight_stats' in all_results[model]['weights']:
                    num_classes = all_results[model]['weights']['num_classes']
                    num_missing = all_results[model]['weights'].get('num_missing_classes', 0)
                    feature_dim = all_results[model]['weights']['feature_dim']
                    weight_range = f"{all_results[model]['weights']['weight_stats']['min']:.3f} to {all_results[model]['weights']['weight_stats']['max']:.3f}"
                    print(f"{model:<50} {num_classes:<8} {num_missing:<8} {feature_dim:<10} {weight_range:<15}")
        
        if args.extract_images:
            print(f"\nImage Representation Summary:")
            print(f"{'Model':<50} {'Orig Imgs':<9} {'Classes':<8} {'Features':<10} {'Valid Classes':<13}")
            print("-" * 95)
            
            for model in successful_models:
                if 'images' in all_results[model]:
                    total_classes = all_results[model]['images'].get('total_classes', 0)
                    original_total_images = all_results[model]['images'].get('total_original_images', 0)
                    feature_dim = all_results[model]['images'].get('feature_dim', 0)
                    valid_classes = all_results[model]['images'].get('num_valid_classes', 
                                  all_results[model]['images'].get('total_classes', 0))
                    print(f"{model:<50} {original_total_images:<9} {total_classes:<8} {feature_dim:<10} {valid_classes:<13}")
    
    if classification_results:
        print(f"\n{'='*80}")
        print("CLASSIFICATION ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"  Classification analyses completed: {len(classification_results)}")
        
        print(f"\nClassification Results Summary:")
        print(f"{'Model':<50} {'Test Acc':<10} {'Weight Conf':<12} {'Image Conf':<12}")
        print("-" * 86)
        
        for model_name, results in classification_results.items():
            test_acc = results['test_accuracy']
            weight_conf = results['weight_confidence']
            image_conf = results['image_confidence']
            
            print(f"{model_name:<50} {test_acc:<10.2f} {weight_conf:<12.3f} {image_conf:<12.3f}")
        
        # Summary statistics
        accuracies = [results['test_accuracy'] for results in classification_results.values()]
        if accuracies:
            print(f"\nClassification Accuracy Statistics:")
            print(f"  Mean accuracy: {np.mean(accuracies):.2f}%")
            print(f"  Std accuracy: {np.std(accuracies):.2f}%")
            print(f"  Min accuracy: {np.min(accuracies):.2f}%")
            print(f"  Max accuracy: {np.max(accuracies):.2f}%")
    
    # Permutation Test Analysis Summary (separate from classification)
    if permutation_results:
        print(f"\n{'='*80}")
        print("PERMUTATION TEST ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"  Permutation test analyses completed: {len(permutation_results)}")
        
        print(f"\nPermutation Test Results Summary:")
        print(f"{'Model':<50} {'Obs Distance':<13} {'P-value':<12} {'Effect Size':<12} {'Significance':<13}")
        print("-" * 102)
        
        for model_name, results in permutation_results.items():
            obs_distance = results['observed_distance']
            p_value = results['permutation_test']['p_value']
            effect_size = results['permutation_test']['effect_size']
            significance = results['permutation_test']['significance']
            
            print(f"{model_name:<50} {obs_distance:<13.4f} {p_value:<12.6f} {effect_size:<12.4f} {significance:<13}")
        
        # Note: Detailed statistics for each model are shown in the individual permutation test results above

if __name__ == "__main__":
    main()