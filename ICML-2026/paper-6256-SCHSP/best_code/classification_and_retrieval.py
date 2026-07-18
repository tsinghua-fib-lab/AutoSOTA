import patch_cifar  # must be first: patches torchvision CIFAR before any other import
from datetime import datetime
import os
import argparse
import json

from config.config import CHECKPOINT_DIR, GLOBAL_SEED, VALID_IMAGE_MODELS, VALID_TEXT_MODELS, VALID_CLASSIFICATION_DATASETS, VALID_RETRIEVAL_DATASETS
from utils.utils import set_global_seed, get_backbone, get_text_encoder
from alignment.aligned_models import MLPAlignedTextModel, ClassificationModel, VLM, Text2ConceptsAlignedTextModel, CCAAlignedImageModel, CCAAlignedTextModel
from alignment.train_aligners import train_mlp_aligner, train_cca_aligner, train_text2concepts_aligner, train_mlp_aligner_sequentially
from evaluation.evaluation_classification_and_retrieval import eval_classification, eval_retrieval



def main():
    parser = argparse.ArgumentParser(description='Train or evaluate MLP aligner from text embeddings to image embedding space')
    
    # Model arguments (can be single values or lists)
    parser.add_argument('--text_models', type=str, nargs='+', required=True,
                        help='Text encoder model name(s) (e.g., clip_vitb32 all-roberta-large-v1)')
    parser.add_argument('--image_models', type=str, nargs='+', required=True,
                        help='Image model name(s) (e.g., clip_vitb32 dinov2_vitb14)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training/evaluation')
    parser.add_argument('--save_dir', type=str, default=CHECKPOINT_DIR,
                        help='Directory to save/load trained model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (auto-selects 2500 for ImageNet-1K models, 500 for others)')
    parser.add_argument('--batch_size', type=str, default=512,
                        help='Batch size for training ("None" for full batch gradient descent)')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='Learning rate (default: 5e-3)')
    parser.add_argument('--architecture', type=str, default='two_layer',
                        choices=['single', 'two_layer'],
                        help='MLP architecture: single (k→d), two_layer (k→4d→d without GLU)')
    parser.add_argument('--force', action='store_true',
                        help='Force training even if alignment weights already exist')
    parser.add_argument('--dataset_img_repr', type=str, nargs='*', default=None,
                        help='Dataset name(s) for class labels (e.g., imagenet1kval, imagenet1k, imagenet21k, etc.). Can be a single dataset or multiple datasets separated by spaces')
    parser.add_argument('--few_shot_samples', type=int, nargs='*', default=None,
                        help='Number of samples per class for few-shot learning. Can be a single value or list matching dataset_img_repr length. Valid values: 1, 2, 4, 8, 16. If not provided, uses all available samples.')
    parser.add_argument('--no_weights', action='store_true',
                        help='Train only on image representation datasets, excluding weights dataset. Requires dataset_img_repr to be not None.')
    parser.add_argument('--sequential_training', action='store_true',
                        help='Use sequential training for few-shot learning (default: False)')
    parser.add_argument('--mode', type=str, default='MLP',
                        choices=['MLP', 'text2concepts', 'CCA'],
                        help='Mode for training')
    parser.add_argument('--use_captions', action='store_true',
                        help='Use image captions for training (only applicable for MLP mode)')
    parser.add_argument('--weight_preprocess', type=str, default=None,
                        choices=['mean', 'attention', 'linear', None],
                        help='Preprocessing mode for weight vectors (only for IMAGENET1K_HEAD_MODELS). "mean": apply mean-based normalization using imagenet1kval mean representation. "attention": apply attention-based preprocessing using per-class image representations.')

    # Evaluation arguments
    # parser.add_argument('--datasets', type=str, nargs='*',
    #                     help='Dataset name(s) for evaluation (e.g., cifar10 cifar100 imagenet1kval)')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'retrieval'],
                        help='Task type: classification or retrieval')
    parser.add_argument('--datasets', type=str, nargs='*', required=True, default=None,
                        help='Dataset name(s) for evaluation (e.g., cifar10 cifar100 eurosat)')
    
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED,
                        help=f'Random seed (default: {GLOBAL_SEED})')

    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for classification logits (default: 1.0). '
                        'Lower = sharper, higher = softer.')

    parser.add_argument('--loss', type=str, default='cosine',
                        choices=['cosine', 'infonce'],
                        help='Loss function for MLP training (default: cosine)')
    parser.add_argument('--infonce_tau', type=float, default=0.07,
                        help='Temperature for InfoNCE loss (default: 0.07)')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability for MLP hidden layer (default: 0.5)')
    parser.add_argument('--input_dropout', type=float, default=0.3,
                        help='Input dropout probability for MLP (default: 0.3)')

    parser.add_argument('--prompt_ensemble', action='store_true',
                        help='Use ensemble of 3 prompt templates (default: single template)')

    parser.add_argument('--post_hoc_standardize', action='store_true',
                        help='Apply I0Tpost embedding standardization at inference')
    
    args = parser.parse_args()

    set_global_seed(args.seed)  # Set a global seed for reproducibility

    if args.task == "retrieval":
        args.sequential_training = False
        
    # Validate no_weights argument
    if args.no_weights and args.dataset_img_repr is None:
        raise ValueError("--dataset_img_repr must be specified when using --no_weights")

    # Handle dataset_img_repr argument - convert single-element list to string for backward compatibility
    if isinstance(args.dataset_img_repr, list) and len(args.dataset_img_repr) == 1:
        args.dataset_img_repr = args.dataset_img_repr[0]
        print(f"📊 Single dataset_img_repr provided: {args.dataset_img_repr}")
    elif isinstance(args.dataset_img_repr, list):
        print(f"📊 Multiple dataset_img_repr provided: {args.dataset_img_repr}")

    # Handle few_shot_samples argument
    if args.few_shot_samples is not None:
        # Validate each few_shot_samples value
        # valid_values = [1, 2, 4, 8, 16]
        # for fs_samples in args.few_shot_samples:
        #     if fs_samples not in valid_values:
        #         raise ValueError(f"few_shot_samples values must be one of {valid_values}, got {fs_samples}")
        
        # Check if length matches dataset_img_repr when both are lists
        if isinstance(args.dataset_img_repr, list) and len(args.few_shot_samples) != len(args.dataset_img_repr):
            raise ValueError(f"few_shot_samples list length ({len(args.few_shot_samples)}) must match dataset_img_repr list length ({len(args.dataset_img_repr)})")
        elif not isinstance(args.dataset_img_repr, list) and len(args.few_shot_samples) != 1:
            raise ValueError(f"When dataset_img_repr is a single value, few_shot_samples must have exactly one value, got {len(args.few_shot_samples)}")
        
        print(f"📊 Few-shot samples configuration: {args.few_shot_samples}")
    else:
        print(f"📊 Using all available samples (no few-shot restriction)")   

    for model in args.text_models:
        assert model in VALID_TEXT_MODELS, f"Invalid text model: {model}. Must be one of {VALID_TEXT_MODELS}"

    for model in args.image_models:
        assert model in VALID_IMAGE_MODELS, f"Invalid image model: {model}. Must be one of {VALID_IMAGE_MODELS}"

    if args.task == 'classification':
        valid_datasets = VALID_CLASSIFICATION_DATASETS
    elif args.task == 'retrieval':
        valid_datasets = VALID_RETRIEVAL_DATASETS
    else:
        raise ValueError(f"Unsupported task: {args.task}. Supported tasks are 'classification' and 'retrieval'.")

    for dataset in args.datasets:
        assert dataset in valid_datasets, f"Invalid dataset: {dataset}. Must be one of {valid_datasets}"

    # mode = 'MLP'  # Default mode
    assert args.mode in ['MLP', 'text2concepts', 'CCA'], "Mode must be either 'MLP', 'text2concepts' or 'cca'"

    if args.mode != 'MLP':
        args.architecture = None

    if not isinstance(args.datasets, list):
        raise ValueError("datasets must be a list of dataset names")

    # Print configuration
    print(f"\n{'='*80}")
    print("MLP ALIGNER TRAINING AND EVALUATION")
    print(f"{'='*80}")
    print(f"Text models: {args.text_models}")
    print(f"Image models: {args.image_models}")
    print(f"Datasets: {args.datasets}")
    print(f"Device: {args.device}")
    print(f"Force retrain: {args.force}")
    print(f"No weights mode: {args.no_weights}")
    
    total_model_combinations = len(args.text_models) * len(args.image_models)
    total_combinations = total_model_combinations * (len(args.datasets))
    print(f"Total model combinations: {total_model_combinations}")
    print(f"Total evaluation combinations: {total_combinations}")
    
    # Initialize results tracking
    training_results = {}
    evaluation_results = {}
    combination_count = 0
    
    # Setup results file with timestamp for incremental saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results_filename = f"mlp_img_aligner_{args.architecture}_{timestamp}.json"
    all_results_path = os.path.join('.', all_results_filename)  # Save to current directory
    
    print(f"📁 Results will be saved incrementally to: {all_results_filename}")
    
    def save_results_incremental():
        """Save all current evaluation results to JSON file."""
        if not args.datasets or not evaluation_results:
            return
            
        # Convert to the same format as evaluation_clip_models.py
        results_list = []
        for eval_combo_key, result in evaluation_results.items():
            if result.get('status') == 'success':
                formatted_result = result
            else:
                formatted_result = {
                    "image_model": result.get('image_model_name', 'unknown'),
                    "text_model": result.get('text_model_name', 'unknown'),
                    "dataset": result.get('dataset_name', 'unknown'),  
                    "status": "failed",
                    "architecture": result.get('architecture', args.architecture),                    
                    "error": result.get('error', 'Unknown error'),
                    "timestamp": result['timestamp']
                }
            results_list.append(formatted_result)
        
        try:
            with open(all_results_path, 'w') as f:
                json.dump(results_list, f, indent=2)
            print(f"💾 Saved {len(results_list)} results to {all_results_filename}")
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
    
    # Iterate through all model combinations
    for image_model_name in args.image_models:
        for text_model_name in args.text_models:
            combination_count += 1
            
            # Normalize image model name
            image_model_normalized = image_model_name.replace('/', '_')
            
            print(f"\n{'='*80}")
            print(f"PROCESSING COMBINATION {combination_count}/{total_model_combinations}")
            print(f"Text model: {text_model_name}")
            print(f"Image model: {image_model_name}")
            print(f"{'='*80}")
            
            # Handle multiple dataset_img_repr for filename
            if isinstance(args.dataset_img_repr, list):
                ds_img_repr_suffix = f"_{'_'.join(args.dataset_img_repr)}"
            elif args.dataset_img_repr is None:
                ds_img_repr_suffix = ""
            else:
                ds_img_repr_suffix = f"_{args.dataset_img_repr}"
            
            # Handle few_shot_samples for filename
            if args.few_shot_samples is not None:
                if isinstance(args.few_shot_samples, list):
                    # Create suffix from list
                    few_shot_suffix = f"_fs{'_'.join(map(str, args.few_shot_samples))}"
                else:
                    few_shot_suffix = f"_fs{args.few_shot_samples}"
            else:
                few_shot_suffix = ""

            setting_suffix = '_sequential' if args.sequential_training else ''
            no_weights_suffix = "_nw" if args.no_weights else ""
            preprocess_suffix = f"_preproc{args.weight_preprocess}" if args.weight_preprocess is not None else ""
            
            if args.mode == 'MLP':
                aligner_filename = f"{args.mode}{setting_suffix}_aligner_{args.architecture}_{image_model_name.replace('/', '_')}_{text_model_name}{ds_img_repr_suffix}{few_shot_suffix}{no_weights_suffix}{preprocess_suffix}.pt"
            else:
                aligner_filename = f"{args.mode}_aligner_{image_model_name.replace('/', '_')}_{text_model_name}{ds_img_repr_suffix}{few_shot_suffix}{no_weights_suffix}{preprocess_suffix}.pt"
            
            aligner_path = os.path.join(args.save_dir, aligner_filename)
            weights_exist = os.path.exists(aligner_path)

            combo_key = f"{text_model_name}_{image_model_normalized}"

            are_both_clip = False
            
            if image_model_name.startswith('clip_') and text_model_name.startswith('clip_'):
                # If both models are CLIP, skip training
                should_train = False
                are_both_clip = True
                print("Both models are CLIP. Skipping MLP aligner training.")
            
            elif weights_exist and not args.force:
                print(f"✅ Alignment weights found: {aligner_path}")
                print("Skipping training. Use --force to retrain.")
                training_results[combo_key] = {
                    'status': 'skipped',
                    'reason': 'weights_exist',
                    'path': aligner_path
                }
                should_train = False
            elif weights_exist and args.force:
                print(f"⚠️  Alignment weights found: {aligner_path}")
                print("🔄 Force flag detected. Will retrain and overwrite existing weights.")
                should_train = True
            else:
                print(f"?? No alignment weights found: {aligner_path}")
                print("🔄 Starting training...")
                should_train = True
            
            # Train if needed
            if should_train:
                print(f"\n{'-'*60}")
                print("TRAINING MLP ALIGNER")
                print(f"{'-'*60}")
                
                try:
                    # Handle batch_size None for full batch gradient descent
                    batch_size = args.batch_size
                    batch_size = int(batch_size)
                    
                    kwargs = {}
                    kwargs['no_weights'] = args.no_weights
                    # Train the MLP aligner
                    if args.mode== 'MLP':
                        if args.sequential_training:
                            train_fnction = train_mlp_aligner_sequentially
                        else:
                            train_fnction = train_mlp_aligner
                        if args.use_captions:
                            kwargs['use_captions'] = True
                        else:
                            kwargs['use_captions'] = False
                        kwargs['loss_type'] = args.loss
                        kwargs['infonce_tau'] = args.infonce_tau
                        kwargs['dropout'] = args.dropout
                        kwargs['input_dropout'] = args.input_dropout
                    elif args.mode== 'text2concepts':
                        if args.sequential_training:
                            raise NotImplementedError(f"Sequential training not implemented for text2concepts.")
                        else:
                            train_fnction = train_text2concepts_aligner
                    elif args.mode== 'CCA':
                        if args.sequential_training:
                            raise NotImplementedError(f"Sequential training not implemented for CCA.")
                        else:
                            train_fnction = train_cca_aligner
                    
                    # Add preprocess to kwargs if specified
                    if args.weight_preprocess is not None:
                        kwargs['preprocess'] = args.weight_preprocess
                        
                    model, aligner_path = train_fnction(
                        text_model_name=text_model_name,
                        image_model_name=image_model_name,
                        device=args.device,
                        num_epochs=args.epochs,
                        batch_size=batch_size,
                        learning_rate=args.lr,
                        save_dir=args.save_dir,
                        architecture=args.architecture,
                        dataset_img_repr=args.dataset_img_repr,
                        few_shot_samples=args.few_shot_samples,
                        filepath=aligner_path,
                        **kwargs
                    )
                    
                    training_results[combo_key] = {
                        'status': 'success',
                        'path': aligner_path,
                        'timestamp': datetime.now().isoformat(),
                        'mode': args.mode,
                        'architecture': args.architecture,
                        'text_model_name': text_model_name,
                        'image_model_name': image_model_name,
                        'no_weights': args.no_weights,
                        'weight_preprocess': args.weight_preprocess
                    }
                    print(f"✅ Training completed successfully for {combo_key}")
                    
                except Exception as e:
                    print(f"❌ Training failed for {combo_key}: {str(e)}")
                    training_results[combo_key] = {
                        'status': 'failed',
                        'error': str(e),
                        'mode': args.mode,
                        'timestamp': datetime.now().isoformat(),
                        'architecture': args.architecture,
                        'text_model_name': text_model_name,
                        'image_model_name': image_model_name,
                        'no_weights': args.no_weights,
                        'weight_preprocess': args.weight_preprocess
                    }
                    continue  # Skip evaluation for this combination
            
            if are_both_clip:
                # Use pre-trained CLIP model directly without an aligner
                print(f"\n{'-'*60}")
                print("LOADING PRE-TRAINED CLIP MODEL FOR EVALUATION")
                print(f"{'-'*60}")
                
                image_encoder, _, _ = get_backbone(image_model_name)
                text_encoder = get_text_encoder(text_model_name, args.device)

            elif args.mode== 'text2concepts':
                image_encoder, _, _ = get_backbone(image_model_name)
                # Use Text2ConceptsTextModel for text model
                text_encoder = Text2ConceptsAlignedTextModel(image_model_name, 
                                                                text_model_name, 
                                                                device='cuda', 
                                                                aligner_path=aligner_path)
            elif args.mode== 'MLP':
                image_encoder, _, _ = get_backbone(image_model_name)
                # Use MLPAlignedTextModel for MLP aligner
                text_encoder = MLPAlignedTextModel(text_model_name, 
                                                        device='cuda', 
                                                        aligner_path=aligner_path)

            elif args.mode== 'CCA':
                image_encoder = CCAAlignedImageModel(image_model_name, 
                                                        device='cuda', 
                                                        aligner_path=aligner_path)
                # Use CCAAlignedTextModel for CCA aligner
                text_encoder = CCAAlignedTextModel(text_model_name, 
                                                        device='cuda', 
                                                        aligner_path=aligner_path)
                
            # Make sure the image_encoder and text_encoder have been correctly loaded
            assert image_encoder is not None, "Image encoder is not loaded."
            assert text_encoder is not None, "Text encoder is not loaded."

            vlm = VLM(image_encoder, text_encoder)

            for dataset in args.datasets:
                eval_combo_key = f"{text_model_name}_{image_model_normalized}_{dataset}"

                print(f"\n📊 Evaluating on dataset: {dataset}")
                
                # try:
                if args.task == 'classification':
                    print("🔍 Evaluating on classification datasets...")
                    # Evaluate the MLP aligner                            
                    if args.prompt_ensemble:
                        prompt_templates = [
                            'A photo of a {}',
                            'A picture of a {}',
                            'An image of a {}'
                        ]
                    else:
                        prompt_templates = None
                    classification_model = ClassificationModel(
                        vlm=vlm,
                        dataset_name=dataset,
                        device=args.device,
                        temperature=args.temperature,
                        prompt_templates=prompt_templates,
                        post_hoc_standardize=args.post_hoc_standardize
                    )

                    results = eval_classification(
                        model=classification_model,
                        dataset_name=dataset,
                        batch_size=args.eval_batch_size,
                        save_dir=args.save_dir,
                    )

                elif args.task == 'retrieval':
                    results = eval_retrieval(
                        model=vlm,
                        dataset_name=dataset,
                        batch_size=args.eval_batch_size,
                        save_dir=args.save_dir,
                    )

                else:
                    raise ValueError(f"Unsupported task: {args.task}. Supported tasks are 'classification' and 'retrieval'.")
                
                if results:
                    results.update({
                        'text_model_name': text_model_name,
                        'image_model_name': image_model_normalized,
                        'architecture': args.architecture,
                        'no_weights': args.no_weights,
                        'status': 'success',
                        'architecture': args.architecture,
                        'weight_preprocess': args.weight_preprocess,
                        'timestamp': datetime.now().isoformat()
                    })
                    evaluation_results[eval_combo_key] = results

                    if args.task == 'classification':
                        # Print classification results
                        print(f"✅ Evaluation in {dataset} successful:")
                        print(f"   Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
                        print(f"   Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
                        print(f"   Total samples: {results['total_samples']}")
                    elif args.task == 'retrieval':
                        # Print retrieval results
                        print(f"✅ Retrieval evaluation in {dataset} successful:")
                        print(f"   i2t mAP: {results['i2t_mAP']*100:.2f}%")
                        print(f"   i2t P@1: {results['i2t_p1']*100:.2f}%")
                        print(f"   i2t P@5: {results['i2t_p5']*100:.2f}%")
                        print(f"   t2i P@1: {results['t2i_p1']*100:.2f}%")

                        # print(f"   Total samples: {results['total_samples']}")

                    # Save incremental results
                    save_results_incremental()
                else:
                    evaluation_results[eval_combo_key] = {
                        'status': 'failed',
                        'error': 'Evaluation returned None',
                        'timestamp': datetime.now().isoformat(),
                        'text_model_name': text_model_name,
                        'image_model_name': image_model_normalized,
                        'dataset_name': dataset,
                        'architecture': args.architecture,
                        'no_weights': args.no_weights
                    }
                    print(f"❌ Evaluation failed for {dataset}")
                    
                    # Save incremental results even for failures
                    save_results_incremental()
                    
            # except Exception as e:
            #     print(f"❌ Evaluation failed for {dataset}: {str(e)}")
            #     evaluation_results[eval_combo_key] = {
            #         'status': 'failed',
            #         'error': str(e),
            #         'timestamp': datetime.now().isoformat(),
            #         'text_model_name': text_model_name,
            #         'image_model_name': image_model_normalized,
            #         'dataset_name': dataset,
            #         'architecture': args.architecture,
            #         'no_weights': args.no_weights
            #     }
                
            #     # Save incremental results even for failures
            #     save_results_incremental()

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    # Training summary
    print(f"\n📈 Training Summary:")
    successful_training = [r for r in training_results.values() if r.get('status') == 'success']
    skipped_training = [r for r in training_results.values() if r.get('status') == 'skipped']
    failed_training = [r for r in training_results.values() if r.get('status') == 'failed']
    
    print(f"   Total model combinations: {len(training_results)}")
    print(f"   Successfully trained: {len(successful_training)}")
    print(f"   Skipped (weights exist): {len(skipped_training)}")
    print(f"   Failed: {len(failed_training)}")
    
    # Evaluation summary
    print(f"\n� Evaluation Summary:")
    successful_evaluations = [r for r in evaluation_results.values() if r.get('status') == 'success']
    failed_evaluations = [r for r in evaluation_results.values() if r.get('status') == 'failed']
    
    print(f"   Total evaluation combinations: {len(evaluation_results)}")
    print(f"   Successful: {len(successful_evaluations)}")
    print(f"   Failed: {len(failed_evaluations)}")
    
    if successful_evaluations and args.task == 'classification':
        print(f"\n🏆 Best Results (Top-1 Accuracy):")
        # Sort by top-1 accuracy
        best_results = sorted(successful_evaluations, key=lambda x: x.get('top1_accuracy', 0), reverse=True)[:10]
        for i, result in enumerate(best_results, 1):
            print(f"   {i}. {result['text_model_name']} -> {result['image_model_name']} on {result['dataset_name']}: "
                    f"Top-1={result['top1_accuracy']:.2f}%, Top-5={result['top5_accuracy']:.2f}%")
    
    print(f"\n📁 All results have been saved incrementally to: {all_results_filename}")
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
