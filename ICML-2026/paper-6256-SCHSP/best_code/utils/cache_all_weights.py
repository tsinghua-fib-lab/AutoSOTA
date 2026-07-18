#!/usr/bin/env python3
"""
Cache all classifier weights for supported models.

This script downloads and caches the classification head weights for all supported
ImageNet models without building similarity matrices. It uses the same weight loading
functions that are used by the similarity and alignment matrix scripts.
"""

import os
import sys
import traceback
from utils.model_utils import load_head_weights
from config.config import TORCHVISION_MODEL_NAMES, HUGGINGFACE_MODEL_NAMES


def cache_imagenet1k_weights():
    """Cache weights for all ImageNet-1K models."""
    print("=" * 60)
    print("CACHING IMAGENET-1K MODEL WEIGHTS")
    print("=" * 60)
    
    # All ImageNet-1K models (torchvision + ConvNext)
    imagenet1k_models = list(TORCHVISION_MODEL_NAMES)
    
    successful = []
    failed = []
    
    for i, model_name in enumerate(imagenet1k_models, 1):
        print(f"\n[{i:2d}/{len(imagenet1k_models)}] Processing {model_name}...")
        
        try:
            # Use load_head_weights to cache both weight and bias
            # Use relative path to weights directory from project root
            weight, bias, num_classes = load_head_weights(model_name, weights_dir="../weights")
            print(f"✅ Cached {model_name}: {weight.shape} weight, {bias.shape} bias, {num_classes} classes")
            successful.append(model_name)
            
        except Exception as e:
            print(f"❌ Failed to cache {model_name}: {e}")
            failed.append((model_name, str(e)))
    
    print(f"\n📊 ImageNet-1K Results:")
    print(f"  ✅ Successfully cached: {len(successful)} models")
    print(f"  ❌ Failed: {len(failed)} models")
    
    if failed:
        print("\nFailed models:")
        for model_name, error in failed:
            print(f"  - {model_name}: {error}")
    
    return successful, failed


def cache_imagenet21k_weights():
    """Cache weights for all ImageNet-21K models."""
    print("\n" + "=" * 60)
    print("CACHING IMAGENET-21K MODEL WEIGHTS")
    print("=" * 60)
    
    successful = []
    failed = []
    
    for i, model_name in enumerate(HUGGINGFACE_MODEL_NAMES, 1):
        print(f"\n[{i:2d}/{len(HUGGINGFACE_MODEL_NAMES)}] Processing {model_name}...")
        
        try:
            # Use load_head_weights to cache both weight and bias
            # Use relative path to weights directory from project root
            weight, bias, num_classes = load_head_weights(model_name, weights_dir="../weights")
            print(f"✅ Cached {model_name}: {weight.shape} weight, {bias.shape} bias, {num_classes} classes")
            successful.append(model_name)
            
        except Exception as e:
            print(f"❌ Failed to cache {model_name}: {e}")
            failed.append((model_name, str(e)))
    
    print(f"\n📊 ImageNet-21K Results:")
    print(f"  ✅ Successfully cached: {len(successful)} models")
    print(f"  ❌ Failed: {len(failed)} models")
    
    if failed:
        print("\nFailed models:")
        for model_name, error in failed:
            print(f"  - {model_name}: {error}")
    
    return successful, failed


def main():
    """Main function to cache all weights."""
    print("🚀 Starting weight caching for all supported models...")
    print(f"Weights will be saved to: {os.path.abspath('../weights')}")
    
    # Create weights directory if it doesn't exist
    os.makedirs('../weights', exist_ok=True)
    
    try:
        # Cache ImageNet-1K weights
        imagenet1k_successful, imagenet1k_failed = cache_imagenet1k_weights()
        
        # Cache ImageNet-21K weights
        imagenet21k_successful, imagenet21k_failed = cache_imagenet21k_weights()
        
        # Summary
        total_successful = len(imagenet1k_successful) + len(imagenet21k_successful)
        total_failed = len(imagenet1k_failed) + len(imagenet21k_failed)
        total_models = total_successful + total_failed
        
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Total models processed: {total_models}")
        print(f"✅ Successfully cached: {total_successful}")
        print(f"❌ Failed: {total_failed}")
        
        if total_failed > 0:
            print(f"\nSuccess rate: {(total_successful/total_models)*100:.1f}%")
        else:
            print("\n🎉 All models successfully cached!")
        
        # List all cached weight files
        print(f"\n📁 Cached weights directory: {os.path.abspath('../weights')}")
        if os.path.exists('../weights'):
            weight_dirs = [d for d in os.listdir('../weights') if os.path.isdir(os.path.join('../weights', d))]
            print(f"   Found {len(weight_dirs)} cached model directories")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
