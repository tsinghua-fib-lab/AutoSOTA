# Script to load the backbone and classifier of the model

import torch
from torch import nn

from config import EMBEDDINGS_DIR
from utils.utils import get_backbone, get_label_names, generate_embeddings, get_clip_tokenizer, get_clip_text_encoder
from dataloaders.datasets_and_dataloaders import get_dataloaders, get_embeddings_dataloaders

import os

torch.autograd.set_detect_anomaly(True)

def load_weights_from_prompts(classifier_layer, label_names, clip_text_encoder, clip_tokenizer, device):
    prompts = []
    for label_name in label_names:
        article = "an" if label_name[0].lower() in "aeiou" else "a"
        prompt = f'a photo of {article} {label_name}'.format(article=article, label_name=label_name)
        prompts.append(prompt)
    # Create embeddings and assign them to the fc1 layer
    with torch.no_grad():
        text_embeddings = clip_text_encoder(clip_tokenizer(prompts).to(device))
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    
    print(f'\tShape of the text embeddings: {text_embeddings.shape}')

    fc1_layer = classifier_layer.fc1
    fc1_layer.weight = nn.Parameter(text_embeddings)
    
    print(f'\tShape of the fc1_layer matrix: {fc1_layer.weight.shape}')    

    return classifier_layer

def get_classifier_with_projection(input_dim, label_names, model_name, device, 
                classifier_weights_path=None, zs=False, bias_term=False):

    class OneLayerClassifier(nn.Module):
        def __init__(self, input_dim, output_dim, bias=False):
            super(OneLayerClassifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, output_dim, bias=bias)
            #~Change dtype of self.fc1.weights to float16
            print(f'> Shape of fc1 weight: {self.fc1.weight.shape}')
            print(f'> Dtype of fc1 weight: {self.fc1.weight.dtype}')

            self.forward = self._forward
     
        def _forward(self,x):
            return self.fc1(x)        

    classifier_layer = OneLayerClassifier(input_dim=input_dim, output_dim=len(label_names), bias=bias_term)
    
    print(f'> Classifier weights')
    if classifier_weights_path is not None:
        full_state_dict = torch.load(classifier_weights_path, map_location=device)
        # Only load 'weight' and 'bias' keys, discard other keys
        filtered_state_dict = {k: v for k, v in full_state_dict.items() if k in ['weight', 'bias']}
        classifier_layer.fc1.load_state_dict(filtered_state_dict)
        print(f'\tClassifier weights loaded from {classifier_weights_path}')
    elif classifier_weights_path is None and zs:
        clip_tokenizer = get_clip_tokenizer()
        clip_text_encoder = get_clip_text_encoder(model_name, device)
        classifier_layer = load_weights_from_prompts(classifier_layer, label_names, clip_text_encoder, clip_tokenizer, device)
        print(f'\tClassifier weights loaded from prompts embeddings. Nº labels: {len(label_names)}')
    else:
        print('\tNo classifier weights loaded. Training classifier from scratch.')
    return classifier_layer
                

def initialization(model_name, batch_size, dataset_name, use_embeddings, 
                   classifier_weights_path, zs, device, class_idx=None, balance_type="undersample", bias_term=False):
        # Get label names - adjust for binary classification if class_idx is specified
        original_label_names = get_label_names(dataset_name)
        if class_idx is not None:
            # Binary classification: target class vs. all others
            target_class_name = original_label_names[class_idx]
            label_names = [f"not_{target_class_name}", target_class_name]
            print(f"Binary classification setup: {label_names}")
        else:
            label_names = original_label_names
            
        ## Initialize model 
        backbone = None
        input_dim = None
        preprocess = None
        
        # Check if we need to load the backbone
        need_backbone = True
        if use_embeddings:
            # Check if all required embeddings exist
            if not os.path.exists(EMBEDDINGS_DIR):
                os.makedirs(EMBEDDINGS_DIR)
            
            split_list = ['train', 'valid', 'test']
            dataset_name_prefix = dataset_name
            split_embedding_paths_dict = {split: os.path.join(EMBEDDINGS_DIR, 
                f"{dataset_name_prefix}_{model_name}_{split}.pt") for split in split_list}
            missing_embeddings = [not os.path.exists(split_embeddings_path) or not os.path.exists(split_embeddings_path.replace('.pt', '_labels.pt'))
                                  for split_embeddings_path in split_embedding_paths_dict.values()]
            
            if not any(missing_embeddings):
                print('> All embeddings found. Skipping backbone loading for efficiency.')
                need_backbone = False
                # Get input dimension from existing embeddings
                sample_embeddings = torch.load(list(split_embedding_paths_dict.values())[0])
                input_dim = sample_embeddings.shape[1]
                print(f'> Input dimension inferred from embeddings: {input_dim}')
        
        # Only load backbone if needed
        if need_backbone:
            print('> Loading backbone model...')
            backbone, input_dim, preprocess = get_backbone(model_name.replace('timm_','timm/'))
            
            # Get dataset
            if not use_embeddings: 
                train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
                        dataset_name, batch_size, transformations=preprocess)
        
        # Create embeddings if use_embeddings and embeddings_path doesn't exist
        if use_embeddings:
            print('> Using embeddings:')
            
            if any(missing_embeddings):
                if backbone is None:
                    print('> Loading backbone model for embedding generation...')
                    backbone, input_dim, preprocess = get_backbone(model_name)
                
                train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
                    dataset_name, batch_size, transformations=preprocess)
                dataloader_list = [train_dataloader, valid_dataloader, test_dataloader]
            
                for i, split in enumerate(split_list):
                    # Set embeddings_path to a default value if none has been provided
                    split_embeddings_path = split_embedding_paths_dict[split]
                    print(f'\tLooking for {split} embeddings at: {split_embeddings_path}.')
                    # Generate embeddings for the split if it doesn't exist                
                    if not os.path.exists(split_embeddings_path) or not os.path.exists(split_embeddings_path.replace('.pt', '_labels.pt')):
                        print(f'\tEmbeddings not found. Generating embeddings for {split}...')                    
                        # Generate labels_path from embeddings_path
                        split_embeddings_labels_path = split_embeddings_path.replace('.pt', '_labels.pt')
                        generate_embeddings(split_embeddings_path, split_embeddings_labels_path, backbone, dataloader_list[i], device, model_name)
                    else:
                        print(f'\tEmbeddings for {split} found.')
                
                # Clear backbone from memory after embedding generation
                del backbone
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                print('> Backbone cleared from memory after embedding generation.')
                
            # Import optimized settings
            from config import NUM_WORKERS, PIN_MEMORY
            train_dataloader, valid_dataloader, test_dataloader = get_embeddings_dataloaders(
                dataset_name, model_name, batch_size, class_idx=class_idx, 
                balance_type=balance_type, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        # Get classifier
        classifier_layer = get_classifier_with_projection(input_dim, label_names, model_name, device,
                                                        classifier_weights_path, zs, bias_term)
        if use_embeddings:
            # Set the first part of the model to identity function
            model = nn.Sequential(
                nn.Identity(),
                classifier_layer
            )
        else:
            model = nn.Sequential(
                backbone,
                classifier_layer
            )
        model.to(device)

        # Freeze backbone parameters
        for param in model[0].parameters():
            param.requires_grad = False

        # Unfreeze classifier parameters
        for param in model[1].parameters():
            param.requires_grad = True

        return model, train_dataloader, valid_dataloader, test_dataloader, preprocess, label_names