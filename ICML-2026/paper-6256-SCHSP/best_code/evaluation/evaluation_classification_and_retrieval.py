from config.config import DATA_DIR, TEST_FLICKR30_CAPTIONS_PATH
from utils.utils import get_preprocess, get_label_names
from utils.flickr30 import evaluate_flickr_retrieval
from utils.coco import evaluate_coco_retrieval
from dataloaders.datasets_and_dataloaders import get_dataloaders
from evaluation.evaluation_clip_models import evaluate_single_model
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os


def eval_classification(model, dataset_name, batch_size=64, save_dir='analysis'):

    image_model_name = model.image_model_name
    text_model_name = model.text_model_name
    print(f"\n{'='*80}")
    print(f"Evaluating Aligner: {text_model_name} -> {image_model_name} on dataset: {dataset_name}")
    print(f"{'='*80}")
    
    preprocess = get_preprocess(image_model_name)
    
    # Get dataset dataloaders
    print(f"Loading {dataset_name} dataset...")
    _, _, test_loader = get_dataloaders(dataset_name, batch_size, preprocess, only_test=True, shuffle=False)

    
    # Evaluate the model using the same function from evaluation_clip_models.py
    print("Starting evaluation...")
    try:        
        # Get number of classes from label names
        label_names = get_label_names(dataset_name)
        num_classes = len(label_names)

        results = evaluate_single_model(
            model=model,
            dataloader=test_loader,
            device=model.device,
            num_classes=num_classes
        )
        
        # Add additional metadata
        results.update({
            'dataset_name': dataset_name,
            'batch_size': batch_size,
        })
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None
    

def eval_retrieval(model, dataset_name, 
                   batch_size=5000, save_dir='analysis'):

    if dataset_name == 'flickr30k':
        labels = 'flickr30k_test'
    elif dataset_name == 'coco':
        # For COCO, we extract captions from images directly, not from a label file
        labels = None
    else:
        raise NotImplementedError(f"Retrieval evaluation not implemented for dataset: {dataset_name}")

    # Get text embeddings
    if labels is not None:
        # For Flickr30k: get prompts from label file
        prompts = get_label_names(labels)
        txt_batch_size = 5000
        text_features_list = []
        for i in range(0, len(prompts), txt_batch_size):
            batch_prompts = prompts[i:i + txt_batch_size]
            batch_features = model.text_encoder(batch_prompts)
            batch_features = F.normalize(batch_features, dim=1)
            text_features_list.append(batch_features)
            print(f"Processed batch {i // txt_batch_size + 1}/{(len(prompts) + txt_batch_size - 1) // txt_batch_size}")
        text_embeddings = torch.cat(text_features_list, dim=0)
    else:
        # For COCO: extract captions from dataloader
        text_embeddings = None

    # Get image embeddings and captions for COCO
    image_model_name = model.image_model_name
    preprocess = get_preprocess(image_model_name)
    _, _, test_loader = get_dataloaders(dataset_name, batch_size, preprocess, only_test=True, shuffle=False)
    
    image_features = []
    all_captions = []
    total_samples = 0
    
    print(f"Batch size for image feature extraction: {batch_size}")
    with torch.no_grad():
        for images, captions in tqdm(test_loader, desc="Extracting image features"):
            images = images.to(model.device)
            feats = model.image_encoder(images)
            feats = F.normalize(feats, p=2, dim=-1)
            image_features.append(feats)
            total_samples += feats.size(0)
            
            # For COCO, collect all captions
            if dataset_name == 'coco':
                all_captions.extend(captions)
    
    image_embeddings = torch.cat(image_features, dim=0)

    # Process text embeddings for COCO
    if dataset_name == 'coco':
        # Flatten all captions from tuples (keep all captions, including duplicates)
        all_captions_flat = []
        
        for image_idx, caption_tuple in enumerate(all_captions):
            for caption in caption_tuple:
                if caption:  # Skip empty captions
                    all_captions_flat.append(caption)
        
        print(f"Extracting text embeddings for {len(all_captions_flat)} captions")
        
        # Extract embeddings for all captions in batches
        text_batch_size = 2000
        text_features_list = []
        for i in range(0, len(all_captions_flat), text_batch_size):
            batch_captions = all_captions_flat[i:i + text_batch_size]
            batch_features = model.text_encoder(batch_captions)
            batch_features = F.normalize(batch_features, dim=1)
            text_features_list.append(batch_features)
            print(f"Processed text batch {i // text_batch_size + 1}/{(len(all_captions_flat) + text_batch_size - 1) // text_batch_size}")
        
        text_embeddings = torch.cat(text_features_list, dim=0)

    # Evaluate retrieval
    if dataset_name == 'flickr30k':
        results = evaluate_flickr_retrieval(image_embeddings, text_embeddings, TEST_FLICKR30_CAPTIONS_PATH)
    elif dataset_name == 'coco':
        coco_json_file = os.path.join(DATA_DIR, 'coco', 'dataset_coco_karpathy_test.json')
        results = evaluate_coco_retrieval(image_embeddings, text_embeddings, coco_json_file)
    else:
        raise NotImplementedError(f"Retrieval evaluation not implemented for dataset: {dataset_name}")

    # Add additional metadata
    results.update({
        'dataset_name': dataset_name,
        'batch_size': batch_size
    })
        
    return results
