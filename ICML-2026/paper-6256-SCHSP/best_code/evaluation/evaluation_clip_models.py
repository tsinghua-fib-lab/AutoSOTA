import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score
from utils.utils import get_backbone, get_text_encoder, get_label_names, get_preprocess
from dataloaders.datasets_and_dataloaders import get_dataloaders


class CLIPEvaluationModel(nn.Module):
    """
    A model that combines CLIP visual backbone with text-encoded class representations
    for zero-shot classification.
    """
    
    def __init__(self, model_name, dataset_name, device):
        super(CLIPEvaluationModel, self).__init__()
        self.device = device
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        # Load the visual backbone
        print(f"Loading visual backbone for {model_name}...")
        self.backbone, _, _ = get_backbone(model_name)
        self.backbone.to(device)
        self.backbone.eval()
        
        # Load the text encoder
        print(f"Loading text encoder for {model_name}...")
        self.text_encoder = get_text_encoder(model_name, device)
        
        # Get class names for the dataset
        print(f"Getting class names for {dataset_name}...")
        class_names = get_label_names(dataset_name)
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Create prompts using the template
        prompts = [f"A photo of a {class_name}" for class_name in class_names]
        print(f"Created {len(prompts)} prompts for {dataset_name}")
        
        # Encode the prompts
        print("Encoding text prompts...")
        with torch.no_grad():
            text_embeddings = self.text_encoder(prompts)
            # Normalize the text embeddings using L2 norm
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
            
        # Store the normalized text embeddings as a parameter (transposed for matrix multiplication)
        # Shape: [embedding_dim, num_classes]
        self.register_buffer('text_embeddings_T', text_embeddings.T)
        
        print(f"Text embeddings shape: {text_embeddings.shape}")
        print(f"Transposed text embeddings shape: {self.text_embeddings_T.shape}")
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        # Get visual features from the backbone
        visual_features = self.backbone(x)
        
        # Normalize visual features using L2 norm
        visual_features = F.normalize(visual_features, p=2, dim=-1)
        
        # Ensure both tensors have the same dtype to avoid dtype mismatch
        # Convert both to float32 for compatibility
        visual_features = visual_features.float()
        text_embeddings_T = self.text_embeddings_T.float()
        
        # Compute similarity scores by matrix multiplication
        # visual_features: [batch_size, embedding_dim]
        # text_embeddings_T: [embedding_dim, num_classes]
        # result: [batch_size, num_classes]
        logits = torch.matmul(visual_features, text_embeddings_T)
        
        return logits


def create_clip_evaluation_model(model_name, dataset_name, device='cuda'):
    """
    Create a CLIP evaluation model for zero-shot classification.
    
    Args:
        model_name (str): Name of the CLIP model (e.g., 'clip_vitb32', 'clip_vitb16', etc.)
        dataset_name (str): Name of the dataset (e.g., 'cifar10', 'cifar100', 'imagenet1kval', etc.)
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        CLIPEvaluationModel: The complete evaluation model
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    device = torch.device(device)
    model = CLIPEvaluationModel(model_name, dataset_name, device)
    return model


def evaluate_single_model(model, dataloader, device, num_classes=None):
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
    
    # Lists to store all predictions and labels for sklearn metrics
    all_preds = []
    all_labels = []

    print(f"Evaluating on dataset with {num_classes} classes...")
    
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
            
            # Store predictions and labels for sklearn metrics
            all_preds.extend(pred_top5[0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
    
    # Calculate sklearn metrics
    balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted') * 100

    # Calculate per-class accuracies
    class_acc_top1 = (class_correct_top1 / (class_total + 1e-8)).numpy()
    class_acc_top5 = (class_correct_top5 / (class_total + 1e-8)).numpy()
    
    print(f"Final Results:")
    print(f"  Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"  Top-5 Accuracy: {top5_accuracy:.2f}%")
    print(f"  Balanced Accuracy: {balanced_acc:.2f}%")
    print(f"  Macro F1: {macro_f1:.2f}%")
    print(f"  Weighted F1: {weighted_f1:.2f}%")
    print(f"  Total samples evaluated: {total_samples}")
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'total_samples': total_samples,
        'class_accuracies_top1': class_acc_top1.tolist(),
        'class_accuracies_top5': class_acc_top5.tolist()
    }


def evaluate_clip_model_on_dataset(model_name, dataset_name, device='cuda', batch_size=64):
    """
    Evaluate a CLIP model on a specific dataset using zero-shot classification.
    
    Args:
        model_name (str): Name of the CLIP model (e.g., 'clip_vitb32', 'clip_vitb16', etc.)
        dataset_name (str): Name of the dataset (e.g., 'cifar10', 'cifar100', etc.)
        device (str): Device to run evaluation on ('cuda' or 'cpu')
        batch_size (int): Batch size for evaluation
        
    Returns:
        Dict containing evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating CLIP model: {model_name} on dataset: {dataset_name}")
    print(f"{'='*80}")
    
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    device = torch.device(device)
    
    # Create the CLIP evaluation model
    print("Creating CLIP evaluation model...")
    try:
        clip_model = create_clip_evaluation_model(model_name, dataset_name, device)
        print(f"Successfully created CLIP model for {dataset_name}")
    except Exception as e:
        print(f"Error creating CLIP model: {e}")
        return None
    
    # Get preprocessing transforms for the model
    print("Getting preprocessing transforms...")
    try:
        preprocess = get_preprocess(model_name)
        print("Successfully obtained preprocessing transforms")
    except Exception as e:
        print(f"Error getting preprocessing transforms for {model_name}: {e}")
        return None
    
    # Get dataset dataloaders
    print(f"Loading {dataset_name} dataset...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(dataset_name, batch_size, preprocess)
        
        # Use test loader if available, otherwise use validation loader
        eval_loader = test_loader if test_loader is not None else val_loader
        
        if eval_loader is None:
            print(f"No evaluation data available for {dataset_name}")
            return None
            
        print(f"Dataset loaded with batch size: {batch_size}")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None
    
    # Evaluate the model
    print("Starting evaluation...")
    try:
        # Get number of classes from label names
        label_names = get_label_names(dataset_name)
        num_classes = len(label_names)
        
        results = evaluate_single_model(
            model=clip_model,
            dataloader=eval_loader,
            device=device,
            model_name=f"{model_name}_{dataset_name}",
            num_classes=num_classes
        )
        
        # Add additional metadata
        results.update({
            'model_name': model_name,
            'dataset_name': dataset_name,
            'batch_size': batch_size,
            'device': str(device),
            'class_names': clip_model.class_names
        })
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None


def save_results_incremental(all_results, dataset_name, filename):
    """
    Save results incrementally to JSON file with error handling.
    
    Args:
        all_results (dict): Dictionary containing results for all models
        dataset_name (str): Name of the dataset evaluated  
        filename (str): Name of the JSON file to save to
    """
    try:
        # Convert results to the desired flat list format
        results_list = []
        
        for combo_key, result in all_results.items():
            if result and result.get('status') == 'success':
                # Convert from percentage (0-100) to decimal (0-1) format
                formatted_result = {
                    "image_model": result['model_name'],
                    "text_model": result['model_name'],  # For CLIP, image and text models are the same
                    "dataset": result['dataset_name'],
                    "top1_accuracy": result['top1_accuracy'] / 100.0,  # Convert percentage to decimal
                    "top5_accuracy": result['top5_accuracy'] / 100.0,  # Convert percentage to decimal
                    "status": "success",
                    "use_definitions": False,  # CLIP uses class names, not definitions
                    "timestamp": result['timestamp']
                }
            else:
                # Handle failed results
                formatted_result = {
                    "image_model": result.get('model_name', 'unknown'),
                    "text_model": result.get('model_name', 'unknown'),
                    "dataset": result.get('dataset_name', 'unknown'),
                    "top1_accuracy": None,
                    "top5_accuracy": None,
                    "status": "failed",
                    "use_definitions": False,
                    "error": result.get('error', 'Unknown error'),
                    "timestamp": result.get('timestamp', datetime.now().isoformat())
                }
            
            results_list.append(formatted_result)
        
        # Save as a simple list (matching the format from evaluation_text2weights.py)
        with open(filename, 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f"💾 Saved {len(results_list)} results to {filename}")
    except Exception as e:
        print(f"⚠️  Failed to save results: {e}")


def load_existing_results(filename):
    """Load existing results from JSON file if it exists."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                results_list = json.load(f)
                # Convert back to dictionary format for internal use
                all_results = {}
                for result in results_list:
                    combo_key = f"{result['image_model']}_{result['dataset']}"
                    # Convert back to percentage format for internal consistency
                    if result.get('status') == 'success' and result['top1_accuracy'] is not None:
                        internal_result = result.copy()
                        internal_result['model_name'] = result['image_model']
                        internal_result['dataset_name'] = result['dataset']
                        internal_result['top1_accuracy'] = result['top1_accuracy'] * 100.0  # Convert back to percentage
                        internal_result['top5_accuracy'] = result['top5_accuracy'] * 100.0  # Convert back to percentage
                        all_results[combo_key] = internal_result
                    else:
                        # Failed result
                        all_results[combo_key] = {
                            'model_name': result['image_model'],
                            'dataset_name': result['dataset'],
                            'status': 'failed',
                            'error': result.get('error', 'Unknown error'),
                            'timestamp': result['timestamp']
                        }
                return all_results
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"⚠️  Could not load existing results from {filename}, starting fresh")
            return {}
    return {}


def test_multiple_clip_models(models_to_test, dataset_names, device='cuda', batch_size=32):
    """
    Test multiple CLIP models on multiple datasets and save results incrementally.
    
    Args:
        models_to_test (list): List of model names to test
        dataset_names (list): List of dataset names to test on
        device (str): Device to use for evaluation
        batch_size (int): Batch size for evaluation
        
    Returns:
        dict: Results for all model-dataset combinations
    """
    print(f"\n{'='*80}")
    print(f"TESTING MULTIPLE CLIP MODELS ON MULTIPLE DATASETS")
    print(f"{'='*80}")
    print(f"Models to test: {models_to_test}")
    print(f"Datasets to test: {dataset_names}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Total combinations: {len(models_to_test)} models × {len(dataset_names)} datasets = {len(models_to_test) * len(dataset_names)}")
    
    # Setup results file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"clip_evaluation_results_{timestamp}.json"
    
    # Load existing results if resuming
    all_results = load_existing_results(results_filename)
    
    if all_results:
        print(f"📂 Resuming evaluation - loaded {len(all_results)} existing results")
    else:
        print(f"🆕 Starting new evaluation - results will be saved to {results_filename}")
    
    # Create set of completed combinations to avoid duplication
    completed_combinations = set()
    for combo_key, result in all_results.items():
        if isinstance(result, dict) and 'model_name' in result and 'dataset_name' in result:
            completed_combinations.add((result['model_name'], result['dataset_name']))
    
    if completed_combinations:
        print(f"⏭️  Skipping {len(completed_combinations)} already completed combinations")
    
    total_combinations = len(models_to_test) * len(dataset_names)
    combination_count = 0
    
    for dataset_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"📊 EVALUATING ON DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        for model_name in models_to_test:
            combination_count += 1
            
            # Create unique key for this model-dataset combination
            combo_key = f"{model_name}_{dataset_name}"
            
            # Skip if already completed
            if (model_name, dataset_name) in completed_combinations:
                print(f"\n⏭️  Skipping already completed: {model_name} on {dataset_name}")
                continue
                
            print(f"\n🔄 Testing combination {combination_count}/{total_combinations}: {model_name} on {dataset_name}")
            print("-" * 60)
            
            try:
                result = evaluate_clip_model_on_dataset(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    device=device,
                    batch_size=batch_size
                )
                
                if result:
                    # Add timestamp and status to result
                    result.update({
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    })
                    all_results[combo_key] = result
                    print(f"✅ SUCCESS: {model_name} on {dataset_name}")
                    print(f"   Top-1: {result['top1_accuracy']:.2f}%")
                    print(f"   Top-5: {result['top5_accuracy']:.2f}%")
                else:
                    # Store failed result
                    all_results[combo_key] = {
                        'model_name': model_name,
                        'dataset_name': dataset_name,
                        'status': 'failed',
                        'error': 'Evaluation returned None',
                        'timestamp': datetime.now().isoformat()
                    }
                    print(f"❌ FAILED: {model_name} on {dataset_name}")
                    
            except Exception as e:
                print(f"\n❌ ERROR with {model_name} on {dataset_name}: {str(e)}")
                # Store failed result with error details
                all_results[combo_key] = {
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Save results after each combination (incremental saving)
            save_results_incremental(all_results, f"multi_{len(dataset_names)}_datasets", results_filename)
    
    # Print summary of all results
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'='*80}")
    
    # Group results by dataset for better readability
    for dataset_name in dataset_names:
        print(f"\nDataset: {dataset_name.upper()}")
        print(f"{'Model':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Status'}")
        print("-" * 60)
        
        for model_name in models_to_test:
            combo_key = f"{model_name}_{dataset_name}"
            result = all_results.get(combo_key)
            if result and result.get('status') == 'success':
                top1 = f"{result['top1_accuracy']:.2f}%"
                top5 = f"{result['top5_accuracy']:.2f}%"
                status = "✅ Success"
            else:
                top1 = "N/A"
                top5 = "N/A" 
                status = "❌ Failed"
                
            print(f"{model_name:<20} {top1:<12} {top5:<12} {status}")
    
    print(f"\n📁 Final results saved to: {results_filename}")
    print(f"{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    
    # Print summary statistics
    successful_evals = [r for r in all_results.values() if r.get('status') == 'success']
    failed_evals = [r for r in all_results.values() if r.get('status') == 'failed']
    
    print(f"\n📈 Evaluation Summary:")
    print(f"   Total combinations: {len(all_results)}")
    print(f"   Successful: {len(successful_evals)}")
    print(f"   Failed: {len(failed_evals)}")
    
    if successful_evals:
        print(f"\n🏆 Best Results (Top-1 Accuracy):")
        # Sort by top-1 accuracy
        best_results = sorted(successful_evals, key=lambda x: x.get('top1_accuracy', 0), reverse=True)[:10]  # Show top 10
        for i, result in enumerate(best_results, 1):
            print(f"   {i}. {result['model_name']} on {result['dataset_name']}: "
                  f"Top-1={result['top1_accuracy']:.2f}%, Top-5={result['top5_accuracy']:.2f}%")
    
    return all_results


# Example usage
if __name__ == "__main__":
    # Test all four CLIP models on multiple datasets
    models_to_test = [
        "clip_vitb32",
        "clip_vitb16", 
        "clip_resnet50",
        "clip_resnet101"
    ]
    
    # Test on multiple datasets
    dataset_names = ["resisc45"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128  # Use reasonable batch size for evaluation
    
    # Run evaluation for all model-dataset combinations
    all_results = test_multiple_clip_models(
        models_to_test=models_to_test,
        dataset_names=dataset_names,
        device=device,
        batch_size=batch_size
    )