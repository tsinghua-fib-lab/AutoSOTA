#!/usr/bin/env python3
"""
Script to extract ImageNet-21k final layer weights and convert them to ImageNet-1k format.
Also includes comprehensive tests to validate the converted layer.

This script:
1. Extracts weights from ImageNet-21k pretrained models
2. Maps them to ImageNet-1k class indices using WNID mappings
3. Creates a new 1000-class linear layer
4. Tests the converted layer on ImageNet-1k validation set
"""

import os
import torch
import torch.nn as nn
import timm
import numpy as np
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from config.config import DATA_DIR

# Import from local utils
try:
    from utils.model_utils import _get_final_linear_layer
    from utils.utils import get_backbone, get_preprocess
except ImportError as e:
    print(f"Warning: Could not import from utils: {e}")
    print("Make sure you're running this script from the alignment directory")
    exit(1)


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)


def read_imagenet1k_wnids(path=None):
    """
    Read the file with ImageNet-1k WNIDs (one per line) and return the list.
    """
    path = path or data_path('imagenet1k', 'imagenet1k_wnids.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"ImageNet-1k WNIDs file not found at '{path}'")
    
    imagenet1k_wnids = []
    with open(path, 'r') as f:
        for line in f:
            wn = line.strip()
            if wn:
                imagenet1k_wnids.append(wn)
    print(f"Read {len(imagenet1k_wnids)} WNIDs from ImageNet-1k from '{path}'.")
    return imagenet1k_wnids

def read_imagenet12k_wnids(path=None):
    """
    Read the file with ImageNet12k WNIDs (one per line) and return the list.
    """
    path = path or data_path('imagenet12k_wnids.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"ImageNet12k WNIDs file not found at '{path}'")
    
    wnids12k = []
    with open(path, 'r') as f:
        for line in f:
            wn = line.strip()
            if wn:
                wnids12k.append(wn)
    print(f"Read {len(wnids12k)} WNIDs from ImageNet12k from '{path}'.")
    return wnids12k


def read_imagenet21k_wnids(path=None):
    """
    Read the file with ImageNet21k WNIDs (one per line) and return the list.
    """
    path = path or data_path('imagenet21k_wnids.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"ImageNet21k WNIDs file not found at '{path}'")
    
    wnids21k = []
    with open(path, 'r') as f:
        for line in f:
            wn = line.strip()
            if wn:
                wnids21k.append(wn)
    print(f"Read {len(wnids21k)} WNIDs from ImageNet21k from '{path}'.")
    return wnids21k

def read_imagenet21k_2extra_wnids12k(path=None):
    """
    Read the file with ImageNet21k 2 extra WNIDs (one per line) and return the list.
    """
    path = path or data_path('imagenet21k_2extra_wnids.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"ImageNet21k 2 extra WNIDs file not found at '{path}'")
    
    wnids21k_2extra = []
    with open(path, 'r') as f:
        for line in f:
            wn = line.strip()
            if wn:
                wnids21k_2extra.append(wn)
    print(f"Read {len(wnids21k_2extra)} WNIDs from ImageNet21k 2 extra from '{path}'.")
    return wnids21k_2extra


def compute_index_mapping(im1k_wnids, im21k_wnids):
    """
    Given list of WNIDs from 1k and 21k, returns:
    - List of indices in im21k_wnids that match each wnid from im1k_wnids (in im1k_wnids order).
    - List of 1k WNIDs that were not found in 21k.
    - Mapping from ImageNet-1k index to ImageNet-21k index (None if missing)
    """
    # Create mapping from WNID to index in the 21k list
    idx_map21k = {wnid: idx for idx, wnid in enumerate(im21k_wnids)}
    indices = []
    missing_wnids = []
    index_mapping = []  # Maps ImageNet-1k index to ImageNet-21k index (None if missing)
    
    for i, wnid in enumerate(im1k_wnids):
        if wnid in idx_map21k:
            idx_21k = idx_map21k[wnid]
            indices.append(idx_21k)
            index_mapping.append(idx_21k)
        else:
            missing_wnids.append(wnid)
            index_mapping.append(None)  # Mark as missing
    
    print(f"\nMapping Results:")
    print(f"- Of the {len(im1k_wnids)} WNIDs from ImageNet Subset, {len(indices)} were found in the 21k list, {len(missing_wnids)} are missing.")
    
    if missing_wnids:
        print(f"\nWARNING: the following {len(missing_wnids)} WNIDs from ImageNet-1k were NOT found in the 21k list:")
        for i, wn in enumerate(missing_wnids):
            print(f"   {i+1:3d}. {wn}")
    else:
        print("\nAll WNIDs from ImageNet-1k were found in the 21k list.")
    
    return indices, missing_wnids, index_mapping


def extract_final_layer_weights(model, model_name):
    """
    Extract weights and biases from the final linear layer of a model.
    
    Args:
        model: PyTorch model with a classification head
        model_name: Name of the model (required for TIMM models)
        
    Returns:
        dict: {'weights': weight_tensor, 'bias': bias_tensor_or_None, 'out_features': int, 'in_features': int}
    """
    final_layer = _get_final_linear_layer(model, model_name)
    
    if not isinstance(final_layer, nn.Linear):
        raise ValueError(f"Expected final layer to be nn.Linear, got {type(final_layer)}")
    
    # Extract weights and biases
    weights = final_layer.weight.data.clone()  # Shape: [out_features, in_features]
    bias = final_layer.bias.data.clone() if final_layer.bias is not None else None  # Shape: [out_features]
    
    print(f"   Final layer weights shape: {weights.shape}")
    print(f"   Final layer bias shape: {bias.shape if bias is not None else None}")
    print(f"   Output features: {final_layer.out_features}")
    print(f"   Input features: {final_layer.in_features}")
    
    return {
        'weights': weights,
        'bias': bias,
        'out_features': final_layer.out_features,
        'in_features': final_layer.in_features
    }


def create_imagenet1k_layer_from_21k(last_layer_info, index_mapping, device='cpu'):
    """
    Create a new Linear layer with 1000 output features using ImageNet-21k weights.
    
    Args:
        last_layer_info: Dictionary from extract_final_layer_weights
        index_mapping: List of length 1000, where index_mapping[i] is the ImageNet-21k index 
                      for ImageNet-1k class i, or None if missing
        device: Device to create the layer on
        
    Returns:
        nn.Linear: New linear layer with 1000 output features
    """
    expected_classes = 21841  # ImageNet-21k typically has 21841 classes
    if last_layer_info['out_features'] != expected_classes:
        print(f"WARNING: Expected {expected_classes} output features for ImageNet-21k, got {last_layer_info['out_features']}")
    
    in_features = last_layer_info['in_features']
    weights_21k = last_layer_info['weights']  # Shape: [21841, in_features]
    bias_21k = last_layer_info['bias']  # Shape: [21841] or None

    # Create new layer with 1000 output features
    new_layer = nn.Linear(in_features, 1000, bias=(bias_21k is not None))
    
    # Initialize with zeros
    nn.init.zeros_(new_layer.weight)
    if new_layer.bias is not None:
        nn.init.zeros_(new_layer.bias)
    
    # Copy weights for available classes
    missing_count = 0
    available_count = 0
    
    for i, idx_21k in enumerate(index_mapping):
        if idx_21k is not None:
            # Copy weights from ImageNet-21k to ImageNet-1k position
            new_layer.weight.data[i] = weights_21k[idx_21k]
            if new_layer.bias is not None and bias_21k is not None:
                new_layer.bias.data[i] = bias_21k[idx_21k]
            available_count += 1
        else:
            # Keep as zeros (already initialized)
            missing_count += 1
    
    print(f"\nCreated ImageNet-1k layer:")
    print(f"   Input features: {in_features}")
    print(f"   Output features: 1000")
    print(f"   Available classes: {available_count}")
    print(f"   Missing classes (zeros): {missing_count}")
    print(f"   Has bias: {new_layer.bias is not None}")
    
    return new_layer.to(device)


def extract_and_convert_imagenet21k_to_1k(model_name, device='cpu'):
    """
    Complete pipeline to extract ImageNet-21k final layer and convert to ImageNet-1k.
    
    Args:
        model_name: Name of the TIMM model (e.g., 'timm/beit_base_patch16_224.in22k_ft_in22k')
        device: Device to load model and create new layer on
        
    Returns:
        tuple: (nn.Linear, dict) - New linear layer and metadata about the conversion
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING IMAGENET-21K TO IMAGENET-1K LAYER for: {model_name}")
    print(f"{'='*80}")
    
    # 1. Load model
    print(f"\n1. Loading model: {model_name}")
    timm_model_name = model_name[5:] if model_name.startswith('timm/') else model_name
    model = timm.create_model(timm_model_name, pretrained=True)
    model.eval()
    model = model.to(device)
    print(f"   ✅ Successfully loaded model")
    
    # 2. Extract final layer weights
    print(f"\n2. Extracting final layer weights...")
    last_layer_info = extract_final_layer_weights(model, model_name)
    
    # 3. Read WNID mappings
    print(f"\n3. Reading WNID mappings...")
    imagenet1k_wnids = read_imagenet1k_wnids()
    imagenet21k_wnids = read_imagenet21k_wnids()
    
    # 4. Compute index mapping
    print(f"\n4. Computing index mapping...")
    indices, missing_wnids, index_mapping = compute_index_mapping(imagenet1k_wnids, imagenet21k_wnids)
    
    # 5. Create new ImageNet-1k layer
    print(f"\n5. Creating ImageNet-1k layer...")
    imagenet1k_layer = create_imagenet1k_layer_from_21k(last_layer_info, index_mapping, device)
    
    # 6. Create metadata
    metadata = {
        'model_name': model_name,
        'original_out_features': last_layer_info['out_features'],
        'in_features': last_layer_info['in_features'],
        'available_classes': len(indices),
        'missing_classes': len(missing_wnids),
        'missing_wnids': missing_wnids,
        'extraction_time': datetime.now().isoformat()
    }
    
    # 7. Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n✅ Successfully created ImageNet-1k layer from ImageNet-21k weights!")
    return imagenet1k_layer, metadata


def create_model_with_new_head(model_name, new_head, device='cpu'):
    """
    Create a model with the new ImageNet-1k head attached.
    
    Args:
        model_name: Name of the TIMM model
        new_head: New linear layer with 1000 outputs
        device: Device to create model on
        
    Returns:
        torch.nn.Module: Complete model with new head
    """
    print(f"\nCreating model with new ImageNet-1k head...")
    
    # Get backbone (feature extractor)
    backbone, features_dim, preprocess = get_backbone(model_name)
    backbone = backbone.to(device)
    backbone.eval()
    
    # Verify dimensions match
    if features_dim != new_head.in_features:
        raise ValueError(f"Dimension mismatch: backbone outputs {features_dim}, head expects {new_head.in_features}")
    
    # Create complete model
    class ModelWithNewHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
            
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
    
    complete_model = ModelWithNewHead(backbone, new_head)
    complete_model = complete_model.to(device)
    complete_model.eval()
    
    print(f"   ✅ Created complete model: backbone({features_dim}) -> head({new_head.out_features})")
    
    return complete_model, preprocess


def test_layer_dimensions(model_name, imagenet1k_layer, device='cpu'):
    """
    Test that the new layer has correct dimensions and can process inputs.
    """
    print(f"\n{'='*60}")
    print(f"TESTING LAYER DIMENSIONS")
    print(f"{'='*60}")
    
    # Get backbone to test dimensions
    backbone, features_dim, _ = get_backbone(model_name)
    backbone = backbone.to(device)
    backbone.eval()
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        # Test backbone output
        backbone_output = backbone(dummy_input)
        print(f"Backbone output shape: {backbone_output.shape}")
        
        # Test new layer
        layer_output = imagenet1k_layer(backbone_output)
        print(f"New layer output shape: {layer_output.shape}")
        
        # Verify shapes
        expected_shape = (2, 1000)
        assert layer_output.shape == expected_shape, f"Expected {expected_shape}, got {layer_output.shape}"
        print(f"✅ Layer dimensions test passed!")
        
        # Check for NaN or Inf
        assert not torch.isnan(layer_output).any(), "Output contains NaN values"
        assert not torch.isinf(layer_output).any(), "Output contains Inf values"
        print(f"✅ Layer output is numerically stable!")
        
        # Print some statistics
        print(f"Output statistics:")
        print(f"   Min: {layer_output.min().item():.4f}")
        print(f"   Max: {layer_output.max().item():.4f}")
        print(f"   Mean: {layer_output.mean().item():.4f}")
        print(f"   Std: {layer_output.std().item():.4f}")


def test_imagenet1k_validation(model_name, imagenet1k_layer, imagenet_path, device='cpu', num_samples=1000):
    """
    Test the converted layer on ImageNet-1k validation set.
    
    Args:
        model_name: Name of the TIMM model
        imagenet1k_layer: Converted linear layer
        imagenet_path: Path to ImageNet dataset
        device: Device for computation
        num_samples: Number of samples to test (None for full validation set)
    """
    print(f"\n{'='*60}")
    print(f"TESTING ON IMAGENET-1K VALIDATION SET")
    print(f"{'='*60}")
    
    if not os.path.exists(imagenet_path):
        print(f"❌ ImageNet path not found: {imagenet_path}")
        print("Skipping ImageNet-1k validation test")
        return None
    
    try:
        # Create complete model with new head
        complete_model, preprocess = create_model_with_new_head(model_name, imagenet1k_layer, device)
        
        # Create validation dataset
        val_dataset = ImageNet(
            root=imagenet_path,
            split='val',
            transform=preprocess
        )
        
        # Limit samples if specified
        if num_samples is not None and num_samples < len(val_dataset):
            indices = torch.randperm(len(val_dataset))[:num_samples]
            val_dataset = torch.utils.data.Subset(val_dataset, indices)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Testing on {len(val_dataset)} validation samples...")
        
        # Run evaluation
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        complete_model.eval()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc="Evaluating")):
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = complete_model(images)
                
                # Top-1 accuracy
                _, pred_top1 = outputs.topk(1, 1, True, True)
                correct_top1 += pred_top1.eq(targets.view(-1, 1)).sum().item()
                
                # Top-5 accuracy
                _, pred_top5 = outputs.topk(5, 1, True, True)
                correct_top5 += pred_top5.eq(targets.view(-1, 1).expand_as(pred_top5)).sum().item()
                
                total += targets.size(0)
                
                # Print progress every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    current_top1 = 100. * correct_top1 / total
                    current_top5 = 100. * correct_top5 / total
                    print(f"   Batch {batch_idx + 1}: Top-1: {current_top1:.2f}%, Top-5: {current_top5:.2f}%")
        
        # Calculate final accuracies
        top1_acc = 100. * correct_top1 / total
        top5_acc = 100. * correct_top5 / total
        
        print(f"\n🎯 ImageNet-1k Validation Results:")
        print(f"   Samples tested: {total}")
        print(f"   Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"   Top-5 Accuracy: {top5_acc:.2f}%")
        
        results = {
            'samples_tested': total,
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'correct_top1': correct_top1,
            'correct_top5': correct_top5
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Error during ImageNet validation test: {e}")
        return None


def test_weight_preservation(original_model_name, imagenet1k_layer, device='cpu'):
    """
    Test that the converted weights match the original model for available classes.
    """
    print(f"\n{'='*60}")
    print(f"TESTING WEIGHT PRESERVATION")
    print(f"{'='*60}")
    
    try:
        # Load original model
        timm_model_name = original_model_name[5:] if original_model_name.startswith('timm/') else original_model_name
        original_model = timm.create_model(timm_model_name, pretrained=True)
        original_model = original_model.to(device)
        original_model.eval()
        
        # Get backbone and original final layer
        backbone, features_dim, _ = get_backbone(original_model_name)
        backbone = backbone.to(device)
        backbone.eval()
        
        original_layer = _get_final_linear_layer(original_model, original_model_name)
        
        # Read mappings to check specific classes
        imagenet1k_wnids = read_imagenet1k_wnids()
        imagenet21k_wnids = read_imagenet21k_wnids()
        indices, missing_wnids, index_mapping = compute_index_mapping(imagenet1k_wnids, imagenet21k_wnids)
        
        # Test a few random available classes
        available_indices = [i for i, idx in enumerate(index_mapping) if idx is not None]
        test_classes = np.random.choice(available_indices, min(10, len(available_indices)), replace=False)
        
        print(f"Testing weight preservation for {len(test_classes)} random classes...")
        
        all_match = True
        for i, class_idx in enumerate(test_classes):
            orig_idx = index_mapping[class_idx]
            
            # Compare weights
            orig_weight = original_layer.weight[orig_idx]
            new_weight = imagenet1k_layer.weight[class_idx]
            
            weight_match = torch.allclose(orig_weight, new_weight, atol=1e-6)
            
            # Compare biases if they exist
            bias_match = True
            if original_layer.bias is not None and imagenet1k_layer.bias is not None:
                orig_bias = original_layer.bias[orig_idx]
                new_bias = imagenet1k_layer.bias[class_idx]
                bias_match = torch.allclose(orig_bias, new_bias, atol=1e-6)
            
            match = weight_match and bias_match
            all_match &= match
            
            status = "✅" if match else "❌"
            print(f"   Class {class_idx:3d} -> Orig {orig_idx:5d}: {status} {'Match' if match else 'Mismatch'}")
        
        if all_match:
            print(f"✅ All tested weights match original model!")
        else:
            print(f"❌ Some weights don't match - there may be an issue with the conversion")
            
        return all_match
        
    except Exception as e:
        print(f"❌ Error during weight preservation test: {e}")
        return False


def save_converted_layer(imagenet1k_layer, metadata, output_path):
    """
    Save the converted layer and metadata to disk.
    """
    print(f"\nSaving converted layer to: {output_path}")
    
    save_data = {
        'layer_state_dict': imagenet1k_layer.state_dict(),
        'metadata': metadata
    }
    
    torch.save(save_data, output_path)
    print(f"✅ Saved successfully!")


def load_converted_layer(input_path, device='cpu'):
    """
    Load a previously converted layer from disk.
    """
    print(f"\nLoading converted layer from: {input_path}")
    
    save_data = torch.load(input_path, map_location=device)
    
    # Recreate the layer
    metadata = save_data['metadata']
    layer = nn.Linear(metadata['in_features'], 1000, bias='bias' in save_data['layer_state_dict'])
    layer.load_state_dict(save_data['layer_state_dict'])
    layer = layer.to(device)
    
    print(f"✅ Loaded layer with {metadata['available_classes']} available classes")
    
    return layer, metadata


def main():
    """Main function to run the extraction and testing pipeline."""
    parser = argparse.ArgumentParser(description='Extract and test ImageNet-1k layer from ImageNet-21k model')
    parser.add_argument('--model', default='timm/beit_base_patch16_224.in22k_ft_in22k', 
                      help='Model name to process')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                      help='Device to use for computation')
    parser.add_argument('--imagenet-path', default=None,
                      help='Path to ImageNet dataset for validation testing')
    parser.add_argument('--num-samples', type=int, default=1000,
                      help='Number of validation samples to test (None for full set)')
    parser.add_argument('--output', default=None,
                      help='Output path to save converted layer')
    parser.add_argument('--skip-validation', action='store_true',
                      help='Skip ImageNet validation testing')
    parser.add_argument('--test-only', default=None,
                      help='Load and test a previously saved layer')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    try:
        if args.test_only:
            # Load and test existing layer
            imagenet1k_layer, metadata = load_converted_layer(args.test_only, device)
            model_name = metadata['model_name']
        else:
            # Extract new layer
            imagenet1k_layer, metadata = extract_and_convert_imagenet21k_to_1k(args.model, device)
            model_name = args.model
            
            # Save if output path specified
            if args.output:
                save_converted_layer(imagenet1k_layer, metadata, args.output)
        
        # Run tests
        print(f"\n{'='*80}")
        print(f"RUNNING TESTS")
        print(f"{'='*80}")
        
        # Test 1: Dimension compatibility
        test_layer_dimensions(model_name, imagenet1k_layer, device)
        
        # Test 2: Weight preservation
        weight_test_passed = test_weight_preservation(model_name, imagenet1k_layer, device)
        
        # Test 3: ImageNet validation (optional)
        validation_results = None
        if not args.skip_validation and args.imagenet_path:
            validation_results = test_imagenet1k_validation(
                model_name, imagenet1k_layer, args.imagenet_path, device, args.num_samples
            )
        
        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Available classes: {metadata['available_classes']}/1000")
        print(f"Missing classes: {metadata['missing_classes']}")
        print(f"Weight preservation: {'✅ PASSED' if weight_test_passed else '❌ FAILED'}")
        
        if validation_results:
            print(f"ImageNet-1k Top-1 Accuracy: {validation_results['top1_accuracy']:.2f}%")
            print(f"ImageNet-1k Top-5 Accuracy: {validation_results['top5_accuracy']:.2f}%")
        
        print(f"\n🎉 Extraction and testing completed!")
        
        # Save results summary
        results_summary = {
            'model_name': model_name,
            'metadata': metadata,
            'weight_preservation_passed': weight_test_passed,
            'validation_results': validation_results,
            'test_time': datetime.now().isoformat()
        }
        
        results_path = f"imagenet1k_layer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"Results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
