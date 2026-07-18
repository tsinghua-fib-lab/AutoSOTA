#!/usr/bin/env python3
"""
ImageNet-1K Evaluation Script for Timm Models

This script evaluates the Timm models listed in IMAGENET21K_HEAD_MODELS on ImageNet-1K validation dataset.
It evaluates each model in three configurations:
1. With original bias terms in the last layer
2. With bias terms zeroed out in the last layer
3. With bias terms zeroed out and weights normalized (L2 norm) in the last layer

The script loads pre-trained models using the same initialization as train.py and evaluates them on the ImageNet-1K validation set.
"""

import torch
import os
import json
import argparse
import traceback
from datetime import datetime
from tqdm import tqdm

from config.config import IMAGENET21K_HEAD_MODELS, IMAGENET21K_2EXTRA_HEAD_MODELS, GLOBAL_SEED
from utils.utils import get_preprocess, set_random_seeds, get_label_names
from utils.model_utils import (
    load_pretrained_model, _get_final_linear_layer, cleanup_model_cache, 
    zero_out_bias_in_last_layer, normalize_weights_in_last_layer, copy_model_state, detect_model_type,
    convert_21k_to_1k_model, prepare_model_for_imagenet1k
)
from dataloaders.datasets_and_dataloaders import get_imagenet1kval_dataloaders

def evaluate_single_model(model, dataloader, device, model_name="unknown", num_classes=None):
    """
    Evaluate a single model on a given dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        model_name: Name of the model for logging
        num_classes: Number of classes in the dataset. If None, will be detected automatically
        
    Returns:
        Dict containing evaluation metrics
    """
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    # Detect number of classes if not provided
    if num_classes is None:
        # Get a sample batch to detect the number of classes from model output
        sample_batch = next(iter(dataloader))
        sample_images = sample_batch[0][:1].to(device)  # Take just one sample
        
        with torch.no_grad():
            sample_output = model(sample_images)
            if hasattr(sample_output, 'logits'):
                sample_output = sample_output.logits
            num_classes = sample_output.shape[1]
        
        print(f"Detected {num_classes} classes from model output")
    else:
        print(f"Using provided number of classes: {num_classes}")
    
    # Class-wise statistics
    class_correct_top1 = torch.zeros(num_classes)
    class_correct_top5 = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    print(f"Evaluating {model_name} on dataset with {num_classes} classes...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluation")):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle different output formats (e.g., HuggingFace models)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            # Get top-5 predictions
            _, pred_top5 = outputs.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()  # Shape: [5, batch_size]
            
            # Check top-1 and top-5 accuracy
            # pred_top5[0] is top-1 predictions, shape: [batch_size]
            correct_top1_batch = pred_top5[0].eq(labels)
            
            # For top-5, we need to check if true labels are in any of the top-5 predictions
            # Expand labels to match pred_top5 shape: [5, batch_size]
            labels_expanded = labels.unsqueeze(0).expand_as(pred_top5)
            correct_top5_batch = pred_top5.eq(labels_expanded).any(dim=0)
            
            # Update totals
            correct_top1 += correct_top1_batch.sum().item()
            correct_top5 += correct_top5_batch.sum().item()
            total_samples += labels.size(0)
            
            # Update class-wise statistics
            for i, label in enumerate(labels):
                class_total[label] += 1
                if correct_top1_batch[i]:
                    class_correct_top1[label] += 1
                if correct_top5_batch[i]:
                    class_correct_top5[label] += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                current_top1 = 100.0 * correct_top1 / total_samples
                current_top5 = 100.0 * correct_top5 / total_samples
                print(f"  Batch {batch_idx + 1}: Top-1: {current_top1:.2f}%, Top-5: {current_top5:.2f}%")
    
    # Calculate final accuracies
    top1_accuracy = 100.0 * correct_top1 / total_samples
    top5_accuracy = 100.0 * correct_top5 / total_samples
    
    # Calculate per-class accuracies
    class_acc_top1 = (class_correct_top1 / (class_total + 1e-8)).numpy()
    class_acc_top5 = (class_correct_top5 / (class_total + 1e-8)).numpy()
    
    print(f"Final Results for {model_name}:")
    print(f"  Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"  Top-5 Accuracy: {top5_accuracy:.2f}%")
    print(f"  Total samples evaluated: {total_samples}")
    
    return {
        'model_name': model_name,
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'total_samples': total_samples,
        'class_accuracies_top1': class_acc_top1.tolist(),
        'class_accuracies_top5': class_acc_top5.tolist()
    }

def evaluate_single_model_imagenet1k(model_name, device='cuda', batch_size=64):
    """
    Evaluate a single model on ImageNet-1K with three configurations:
    1. With original bias terms
    2. With bias terms zeroed out
    3. With bias terms zeroed out and weights normalized (L2 norm)
    Automatically converts ImageNet-21k models to ImageNet-1k if needed.
    
    Args:
        model_name: Name of the model to evaluate
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        Dict containing results for all three configurations
    """
    print(f"\n{'='*80}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*80}")
    
    # Load complete pre-trained model
    print("Loading pre-trained model...")
    try:
        model = load_pretrained_model(model_name)
        print(f"Successfully loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None
    
    # Prepare model for ImageNet-1k evaluation (convert if needed)
    print("Preparing model for ImageNet-1k evaluation...")
    try:
        model, model_type, conversion_applied = prepare_model_for_imagenet1k(model, model_name, device)
        print(f"Model preparation complete: {model_type}")
    except Exception as e:
        print(f"Error preparing model {model_name}: {e}")
        print("This might be due to missing WNID mapping files or unsupported model architecture")
        return None
    
    # Get appropriate preprocessing transforms
    print("Getting preprocessing transforms...")
    try:
        preprocess = get_preprocess(model_name)
        print("Successfully obtained preprocessing transforms")
    except Exception as e:
        print(f"Error getting preprocessing transforms for {model_name}: {e}")
        return None
    
    # Get ImageNet-1K validation dataloader
    print("Loading ImageNet-1K validation dataset...")
    try:
        # For evaluation, we only need the test split (which is validation in our case)
        _, _, test_dataloader = get_imagenet1kval_dataloaders(batch_size, preprocess, only_test=True)
        print(f"Dataset loaded with batch size: {batch_size}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure ImageNet-1K validation data is available under DATA_DIR/imagenet1k/val/")
        return None
    
    results = {
        'model_type': model_type,
        'conversion_applied': conversion_applied
    }
    
    # Evaluation 1: With original bias terms
    print(f"\n{'-'*60}")
    print("Evaluation 1: With original bias terms")
    print(f"{'-'*60}")
    
    # Create a copy of the model for evaluation with bias
    model_with_bias = copy_model_state(model)
    model_with_bias.to(device)
    
    try:
        last_linear = _get_final_linear_layer(model_with_bias, model_name)
        if last_linear.bias is not None:
            print(f"  Original bias stats: min={last_linear.bias.min().item():.4f}, max={last_linear.bias.max().item():.4f}, mean={last_linear.bias.mean().item():.4f}")
        else:
            print("  Model has no bias in last layer")
    except Exception as e:
        print(f"  Could not analyze bias: {e}")
    
    result_with_bias = evaluate_single_model(
        model_with_bias, test_dataloader, device, 
        model_name=model_name, num_classes=1000
    )
    results['with_bias'] = result_with_bias
    
    # Clean up first model
    del model_with_bias
    cleanup_model_cache()
    
    # Evaluation 2: With bias terms zeroed out
    print(f"\n{'-'*60}")
    print("Evaluation 2: With bias terms zeroed out")
    print(f"{'-'*60}")
    
    # Create another copy and zero out the bias
    model_without_bias = copy_model_state(model)
    model_without_bias = zero_out_bias_in_last_layer(model_without_bias, model_name)
    model_without_bias.to(device)
    
    result_without_bias = evaluate_single_model(
        model_without_bias, test_dataloader, device,
        model_name=f"{model_name}_without_bias", num_classes=1000
    )
    results['without_bias'] = result_without_bias
    
    # Clean up second model
    del model_without_bias
    cleanup_model_cache()
    
    # Evaluation 3: With bias zeroed out and weights normalized
    print(f"\n{'-'*60}")
    print("Evaluation 3: With bias zeroed out and weights normalized (L2)")
    print(f"{'-'*60}")
    
    # Create another copy, zero out bias, and normalize weights
    model_normalized = copy_model_state(model)
    model_normalized = zero_out_bias_in_last_layer(model_normalized, model_name)
    model_normalized = normalize_weights_in_last_layer(model_normalized, model_name)
    model_normalized.to(device)
    
    result_normalized = evaluate_single_model(
        model_normalized, test_dataloader, device,
        model_name=f"{model_name}_normalized", num_classes=1000
    )
    results['normalized'] = result_normalized
    
    # Clean up third model
    del model_normalized
    del model
    cleanup_model_cache()
    
    # Print comparison
    print(f"\n{'-'*60}")
    print("Comparison Summary:")
    print(f"{'-'*60}")
    print(f"Model: {model_name}")
    print(f"With bias       - Top-1: {result_with_bias['top1_accuracy']:.2f}%, Top-5: {result_with_bias['top5_accuracy']:.2f}%")
    print(f"Without bias    - Top-1: {result_without_bias['top1_accuracy']:.2f}%, Top-5: {result_without_bias['top5_accuracy']:.2f}%")
    print(f"Normalized      - Top-1: {result_normalized['top1_accuracy']:.2f}%, Top-5: {result_normalized['top5_accuracy']:.2f}%")
    print(f"Bias vs No-bias - Top-1: {result_with_bias['top1_accuracy'] - result_without_bias['top1_accuracy']:.2f}%, Top-5: {result_with_bias['top5_accuracy'] - result_without_bias['top5_accuracy']:.2f}%")
    print(f"No-bias vs Norm - Top-1: {result_without_bias['top1_accuracy'] - result_normalized['top1_accuracy']:.2f}%, Top-5: {result_without_bias['top5_accuracy'] - result_normalized['top5_accuracy']:.2f}%")
    print(f"Bias vs Norm    - Top-1: {result_with_bias['top1_accuracy'] - result_normalized['top1_accuracy']:.2f}%, Top-5: {result_with_bias['top5_accuracy'] - result_normalized['top5_accuracy']:.2f}%")
    
    return results


def test_model_conversion():
    """
    Test the model conversion functionality on a single model.
    """
    print("Testing model conversion functionality...")
    
    # Test with a known ImageNet-21k model
    test_model_name = 'timm/beit_base_patch16_224.in22k_ft_in22k'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Testing with model: {test_model_name}")
    print(f"Device: {device}")
    
    try:
        # Load model
        print("1. Loading pre-trained model...")
        model = load_pretrained_model(test_model_name)
        
        # Check original type
        print("2. Detecting original model type...")
        original_type, original_features = detect_model_type(model, test_model_name)
        print(f"   Original: {original_type} ({original_features} features)")
        
        # Test conversion
        if original_type == 'imagenet21k':
            print("3. Converting ImageNet-21k to ImageNet-1k...")
            converted_model = convert_21k_to_1k_model(model, test_model_name, device)
            
            # Verify conversion
            print("4. Verifying conversion...")
            new_type, new_features = detect_model_type(converted_model, test_model_name)
            print(f"   Converted: {new_type} ({new_features} features)")
            
            if new_features == 1000:
                print("✅ Conversion successful!")
                
                # Test with dummy input
                print("5. Testing with dummy input...")
                converted_model.to(device)
                converted_model.eval()
                
                # Create dummy input
                dummy_input = torch.randn(2, 3, 224, 224).to(device)
                
                with torch.no_grad():
                    output = converted_model(dummy_input)
                    print(f"   Output shape: {output.shape}")
                    
                    if output.shape[1] == 1000:
                        print("✅ Model produces correct output shape for ImageNet-1k!")
                        return True
                    else:
                        print(f"❌ Unexpected output shape: {output.shape}")
                        return False
            else:
                print(f"❌ Conversion failed - still has {new_features} features")
                return False
        elif original_type == 'imagenet1k':
            print("Model already has ImageNet-1k head - no conversion needed")
            print("✅ Test passed (no conversion required)")
            return True
        else:
            print(f"❌ Unknown model type: {original_type}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
        return False
    finally:
        cleanup_model_cache()

def test_evaluation_pipeline():
    """
    Test the complete evaluation pipeline on a single model with small batch.
    """
    print("\nTesting complete evaluation pipeline...")
    
    # Use a smaller model for faster testing
    test_model_name = 'timm/beit_base_patch16_224.in22k_ft_in22k'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4  # Small batch for testing
    
    print(f"Testing with model: {test_model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    try:
        # Test the evaluation function (but only evaluate a few batches)
        print("Testing model evaluation...")
        
        # Load and prepare model
        model = load_pretrained_model(test_model_name)
        model, model_type, conversion_applied = prepare_model_for_imagenet1k(model, test_model_name, device)
        model.to(device)
        
        print(f"Model prepared: {model_type}, conversion applied: {conversion_applied}")
        
        # Get preprocessing
        preprocess = get_preprocess(test_model_name)
        
        # Test with synthetic data instead of real dataset for testing
        print("Creating synthetic test data...")
        
        # Create a simple dataset for testing
        test_images = torch.randn(batch_size, 3, 224, 224)
        test_labels = torch.randint(0, 1000, (batch_size,))
        
        # Create a simple dataloader-like structure
        test_data = [(test_images, test_labels)]
        
        # Test evaluation function with synthetic data
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_data:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                # Get top-5 predictions
                _, pred_top5 = outputs.topk(5, 1, True, True)
                pred_top5 = pred_top5.t()
                
                # Check predictions
                correct_top1_batch = pred_top5[0].eq(labels)
                labels_expanded = labels.unsqueeze(0).expand_as(pred_top5)
                correct_top5_batch = pred_top5.eq(labels_expanded).any(dim=0)
                
                correct_top1 += correct_top1_batch.sum().item()
                correct_top5 += correct_top5_batch.sum().item()
                total_samples += labels.size(0)
        
        print(f"Test evaluation completed:")
        print(f"  Samples processed: {total_samples}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Top-1 correct: {correct_top1}")
        print(f"  Top-5 correct: {correct_top5}")
        
        if outputs.shape[1] == 1000:
            print("✅ Evaluation pipeline test passed!")
            return True
        else:
            print(f"❌ Unexpected output dimensions: {outputs.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        traceback.print_exc()
        return False
    finally:
        cleanup_model_cache()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Timm models on ImageNet-1K validation dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run evaluation on (default: cuda if available)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='Specific models to evaluate (default: all IMAGENET21K_2EXTRA_HEAD_MODELS + IMAGENET21K_HEAD_MODELS)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save results (default: ./evaluation_results)')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED,
                        help=f'Random seed (default: {GLOBAL_SEED})')
    parser.add_argument('--test_conversion', action='store_true',
                        help='Test the model conversion functionality and exit')
    parser.add_argument('--test_pipeline', action='store_true',
                        help='Test the complete evaluation pipeline and exit')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seeds(args.seed)
    
    # Run tests if requested
    if args.test_conversion:
        print("Running model conversion test...")
        success = test_model_conversion()
        exit(0 if success else 1)
    
    if args.test_pipeline:
        print("Running evaluation pipeline test...")
        success = test_evaluation_pipeline()
        exit(0 if success else 1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which models to evaluate
    if args.models:
        models_to_evaluate = args.models
        print(f"Evaluating specified models: {models_to_evaluate}")
    else:
        # models_to_evaluate = IMAGENET21K_2EXTRA_HEAD_MODELS + IMAGENET21K_HEAD_MODELS
        models_to_evaluate =  [IMAGENET21K_2EXTRA_HEAD_MODELS[0]]
        print(f"Evaluating all IMAGENET21K_2EXTRA_HEAD_MODELS + IMAGENET21K_HEAD_MODELS: {len(models_to_evaluate)} models")
    
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Store all results
    all_results = {}
    
    # Evaluate each model
    for i, model_name in enumerate(models_to_evaluate):
        print(f"\n\nProcessing model {i+1}/{len(models_to_evaluate)}: {model_name}")
        
        try:
            results = evaluate_single_model_imagenet1k(model_name, args.device, args.batch_size)
            if results:
                all_results[model_name] = results
                print(f"✅ Successfully evaluated {model_name}")
            else:
                print(f"❌ Failed to evaluate {model_name}")
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {'error': str(e)}
        
        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = os.path.join(args.output_dir, f"imagenet1k_evaluation_intermediate_{timestamp}.json")
        with open(intermediate_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"imagenet1k_evaluation_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    # Print summary
    successful_models = [m for m, r in all_results.items() if 'error' not in r]
    failed_models = [m for m, r in all_results.items() if 'error' in r]
    
    print(f"\nSummary:")
    print(f"  Total models: {len(models_to_evaluate)}")
    print(f"  Successful: {len(successful_models)}")
    print(f"  Failed: {len(failed_models)}")
    
    if failed_models:
        print(f"\nFailed models:")
        for model in failed_models:
            print(f"  - {model}")
    
    if successful_models:
        print(f"\nTop-1 Accuracy Summary:")
        print(f"{'Model':<50} {'Bias':<8} {'No-Bias':<8} {'Normalized':<10} {'B-NB':<8} {'NB-N':<8} {'B-N':<8}")
        print("-" * 110)
        
        for model in successful_models:
            if 'with_bias' in all_results[model] and 'without_bias' in all_results[model] and 'normalized' in all_results[model]:
                acc_bias = all_results[model]['with_bias']['top1_accuracy']
                acc_nobias = all_results[model]['without_bias']['top1_accuracy']
                acc_norm = all_results[model]['normalized']['top1_accuracy']
                diff_bias_nobias = acc_bias - acc_nobias
                diff_nobias_norm = acc_nobias - acc_norm
                diff_bias_norm = acc_bias - acc_norm
                print(f"{model:<50} {acc_bias:<8.2f} {acc_nobias:<8.2f} {acc_norm:<10.2f} {diff_bias_nobias:<8.2f} {diff_nobias_norm:<8.2f} {diff_bias_norm:<8.2f}")

if __name__ == "__main__":
    main()
