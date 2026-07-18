#!/usr/bin/env python3
"""
Train an MLP to align ImageNet-1k class names to text embeddings from various text encoders.
The MLP learns a mapping from class name indices to text embeddings.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# Import utility functions
from utils.utils import get_label_names, get_text_encoder, get_backbone
# Import CCA class
from other_methods.csa.cca_class import NormalizedCCA
      

def get_mlp_aligner_path(dataset_img_repr, 
                         few_shot_samples, architecture, 
                         image_model_name, text_model_name, save_dir, no_weights=False,
                         mode='imgweights', preprocess=None):
    
    # Handle multiple dataset_img_repr for filename
    if isinstance(dataset_img_repr, list):
        ds_img_repr_suffix = f"_{'_'.join(dataset_img_repr)}"
    elif dataset_img_repr is None:
        ds_img_repr_suffix = ""
    else:
        ds_img_repr_suffix = f"_{dataset_img_repr}"
    
    # Handle few_shot_samples for filename
    if few_shot_samples is not None:
        if isinstance(few_shot_samples, list):
            # Create suffix from list, filtering out None values
            fs_values = [str(fs) if fs is not None else "all" for fs in few_shot_samples]
            few_shot_suffix = f"_fs{'_'.join(fs_values)}"
        else:
            few_shot_suffix = f"_fs{few_shot_samples}"
    else:
        few_shot_suffix = ""
    
    no_weights_suffix = "_nw" if no_weights else ""
    preprocess_suffix = f"_preproc{preprocess}" if preprocess is not None else ""
    mlp_filename = f"{mode}_aligner_{architecture}_{image_model_name.replace('/', '_')}_{text_model_name}{ds_img_repr_suffix}{few_shot_suffix}{no_weights_suffix}{preprocess_suffix}.pt"
    mlp_aligner_path = os.path.join(save_dir, mlp_filename)
    print(f"MLP aligner path: {mlp_aligner_path}")
    return mlp_aligner_path

class TextToImageMLP(nn.Module):
    """MLP that maps from text embeddings to image embedding space.
    
    Supports five configurations:
    1) 'single': Single layer linear (k → d)
    4) 'two_layer': Two layer MLP without GLU (k → 4d → d)
    """
    
    def __init__(self, text_embedding_dim, image_embedding_dim, architecture='single', low_rank_dim=None, dropout=0.5, input_dropout_prob=0.3):
        super(TextToImageMLP, self).__init__()
        # Input dropout parameters
        self.input_dropout_prob = input_dropout_prob  # Probability of applying input dropout
        self.architecture = architecture
          
        if architecture in ['single']:
            # Single linear layer MLP: k → d
            self.mlp = nn.Linear(text_embedding_dim, image_embedding_dim, bias=False)

        elif architecture in ['two_layer']:
            # Two layer MLP: k → 4d → d
            # W1 ∈ R^(k × 4d), W2 ∈ R^(4d × d)
            hidden_dim = 4 * image_embedding_dim  # 4d
            self.W1 = nn.Linear(text_embedding_dim, hidden_dim)  # k → 4d
            self.W2 = nn.Linear(hidden_dim, image_embedding_dim)  # 4d → d
            
            # GELU activation and dropout
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(image_embedding_dim)
            

        else:
            raise ValueError(f"Unknown architecture: {architecture}. Choose from 'single', 'single_lbfgs', 'two_layer_glu', 'two_layer', 'lor_two_layer'")
        
    def forward(self, text_embeddings):
        # Ensure input is float32 to avoid dtype issues
        text_embeddings = text_embeddings.float()
        
        # Apply input dropout: randomly drop 50% of input features with probability 0.3
        if self.training and torch.rand(1).item() < self.input_dropout_prob:
            # Create a mask that randomly selects 50% of features to keep
            mask = torch.rand_like(text_embeddings) > 0.5
            text_embeddings = text_embeddings * mask.float()
        
        if self.architecture in ['single']:
            # Single layer MLP path
            mlp_output = self.mlp(text_embeddings)

        elif self.architecture in ['two_layer']:
            # Two layer MLP path: k → 4d → d
            # First layer: k → 4d with GELU activation
            hidden = self.W1(text_embeddings)  # k → 4d
            hidden = self.gelu(hidden)  # GELU activation
            hidden = self.dropout(hidden)  # Dropout with p=0.5
            
            # Second layer: 4d → d
            mlp_output = self.W2(hidden)  # 4d → d
            
            # Apply layer normalization
            mlp_output = self.layer_norm(mlp_output)
            
        # Ensure output is float32
        return mlp_output.float()


class Text2ConceptsAlignedTextModel(nn.Module):
    def __init__(self, image_model_name, text_model_name, device='cuda', aligner_path=None):
        super(Text2ConceptsAlignedTextModel, self).__init__()
        self.device = device
        # self.image_model_name = image_model_name
        self.model_name = text_model_name
        
        # Load the text encoder
        print(f"Loading text encoder for {text_model_name}...")
        self.text_encoder = get_text_encoder(text_model_name, device)

        # Load the trained MLP aligner
        print(f"Loading text2concept aligner from {aligner_path}...")
        checkpoint = torch.load(aligner_path, map_location=device)

        # Get dimensions and architecture from checkpoint
        image_embedding_dim = checkpoint['model_config']['image_embedding_dim']
        clip_embedding_dim = checkpoint['model_config']['text_embedding_dim']        
        
        # Create and load the MLP aligner with the correct architecture
        self.mlp_aligner = TextToImageMLP(clip_embedding_dim, image_embedding_dim, architecture='single')
        self.mlp_aligner.load_state_dict(checkpoint['model_state_dict'])
        self.mlp_aligner.to(device)
        self.mlp_aligner.eval()

    def forward(self, txt):
        # Encode the prompts with text encoder and align them with MLP
        print("Encoding text prompts and aligning to image space...")
        with torch.no_grad():
            # Get text embeddings
            text_embeddings = self.text_encoder(txt)

            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

            # Align text embeddings to image space using trained MLP
            aligned_text_embeddings = self.mlp_aligner(text_embeddings)
            
            # Normalize the aligned text embeddings using L2 norm
            aligned_text_embeddings = F.normalize(aligned_text_embeddings, p=2, dim=-1)

        return aligned_text_embeddings

class MLPAlignedTextModel(nn.Module):
    def __init__(self, model_name, device='cuda', aligner_path=None):
        super(MLPAlignedTextModel, self).__init__()
        self.device = device
        self.model_name = model_name
        
        # Load the text encoder
        print(f"Loading text encoder for {model_name}...")
        self.text_encoder = get_text_encoder(model_name, device)

        # Load the trained MLP aligner
        print(f"Loading MLP aligner from {aligner_path}...")
        checkpoint = torch.load(aligner_path, map_location=device)

        # Get dimensions from checkpoint
        text_embedding_dim = checkpoint['model_config']['text_embedding_dim']
        image_embedding_dim = checkpoint['model_config']['image_embedding_dim']
        architecture = checkpoint['model_config'].get('architecture', 'single')  # Default to 'single' for backward compatibility
        
        # Create and load the MLP aligner
        self.mlp_aligner = TextToImageMLP(text_embedding_dim, image_embedding_dim, architecture=architecture)
        self.mlp_aligner.load_state_dict(checkpoint['model_state_dict'])
        self.mlp_aligner.to(device)
        self.mlp_aligner.eval()

    def forward(self, txt):
        # Encode the prompts with text encoder and align them with MLP
        print("Encoding text prompts and aligning to image space...")
        with torch.no_grad():
            # Get text embeddings
            text_embeddings = self.text_encoder(txt)

            # Text embeddings MUST be normalized - MLP was trained on normalized inputs
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
            # Align text embeddings to image space using trained MLP
            aligned_text_embeddings = self.mlp_aligner(text_embeddings)

            # Normalize the aligned text embeddings using L2 norm
            aligned_text_embeddings = F.normalize(aligned_text_embeddings, p=2, dim=-1)
            aligned_text_embeddings = aligned_text_embeddings.float()  # Ensure float32

        return aligned_text_embeddings

class CCAAlignedTextModel(nn.Module):
    def __init__(self, text_model_name, device='cuda', aligner_path=None):
        super(CCAAlignedTextModel, self).__init__()
        self.device = device
        self.model_name = text_model_name
        
        # Load the text encoder
        print(f"Loading text encoder for {text_model_name}...")
        self.text_encoder = get_text_encoder(text_model_name, device)

        # Instantiate CCA
        self.cca = NormalizedCCA()        

        # Load the trained CCA aligner
        if aligner_path is not None:
            self.cca.load_model(aligner_path)
            print(f"Loading CCA aligner from {aligner_path}...")
        else:
            raise ValueError("CCA aligner path must be provided for CCAAlignedTextModel")
        self.image_embedding_dim = self.cca.dim1

    def forward(self, txt):
        # Encode the prompts with text encoder and align them with CCA
        print("Encoding text prompts and aligning to image space...")
        with torch.no_grad():
            # Get text embeddings
            text_embeddings = self.text_encoder(txt)
            
            # Convert to numpy for CCA transformation
            text_embeddings_np = text_embeddings.cpu().numpy()
            
            # Create dummy image embeddings (zeros) as placeholder for CCA transform
            dummy_image_embeddings = np.zeros((text_embeddings_np.shape[0], self.image_embedding_dim))
            
            # Apply CCA transformation - get the text-aligned embeddings (second output)
            _, aligned_text_embeddings_np = self.cca.transform_data(dummy_image_embeddings, text_embeddings_np)
            
            # Convert back to tensor
            aligned_text_embeddings = torch.from_numpy(aligned_text_embeddings_np).float().to(self.device)
            
            # Normalize the aligned text embeddings using L2 norm
            aligned_text_embeddings = F.normalize(aligned_text_embeddings, p=2, dim=-1)

        return aligned_text_embeddings

class CCAAlignedImageModel(nn.Module):
    def __init__(self, image_model_name, device='cuda', aligner_path=None):
        super(CCAAlignedImageModel, self).__init__()
        self.device = device
        self.model_name = image_model_name
        
        # Load the image encoder
        print(f"Loading image encoder for {image_model_name}...")
        self.image_encoder, _, _ = get_backbone(image_model_name)
        self.image_encoder.to(device)
        self.image_encoder.eval()

        # Instantiate CCA
        self.cca = NormalizedCCA()

        # Load the trained CCA aligner
        if aligner_path is not None:
            self.cca.load_model(aligner_path)
            print(f"Loading CCA aligner from {aligner_path}...")
        else:
            raise ValueError("CCA aligner path must be provided for CCAAlignedImageModel")
        
        self.text_embedding_dim = self.cca.dim2

    def forward(self, images):
        # Encode the images with image encoder and align them with CCA
        print("Encoding images and aligning to text space...")
        with torch.no_grad():
            # Get image embeddings
            image_embeddings = self.image_encoder(images)
            
            # Normalize image embeddings
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            
            # Convert to numpy for CCA transformation
            image_embeddings_np = image_embeddings.cpu().numpy()
            
            # Create dummy text embeddings (zeros) as placeholder for CCA transform
            dummy_text_embeddings = np.zeros((image_embeddings_np.shape[0], self.text_embedding_dim))

            # Apply CCA transformation - get the image-aligned embeddings (first output)
            aligned_image_embeddings_np, _ = self.cca.transform_data(image_embeddings_np, dummy_text_embeddings)
            
            # Convert back to tensor
            aligned_image_embeddings = torch.from_numpy(aligned_image_embeddings_np).float().to(self.device)
            
            # Normalize the aligned image embeddings using L2 norm
            aligned_image_embeddings = F.normalize(aligned_image_embeddings, p=2, dim=-1)

        return aligned_image_embeddings

class VLM():
    def __init__(self, image_encoder, text_encoder, device=None):
        super(VLM, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_model_name = getattr(image_encoder, 'model_name', None)
        self.text_model_name = getattr(text_encoder, 'model_name', None)
        # Set device for both encoders
        if isinstance(self.image_encoder, nn.Module):
            self.image_encoder.to(self.device)
        if isinstance(self.text_encoder, nn.Module):
            self.text_encoder.to(self.device)

    def encode_text(self, texts):
        with torch.no_grad():
            text_embeddings = self.text_encoder(texts)
            return text_embeddings

    def encode_image(self, images):
        with torch.no_grad():
            image_embeddings = self.image_encoder(images)
            return image_embeddings



class ClassificationModel(nn.Module):
    def __init__(self, vlm, dataset_name, device='cuda', temperature=1.0, prompt_templates=None, post_hoc_standardize=False):
        super(ClassificationModel, self).__init__()
        self.vlm = vlm
        self.device = device
        # Get class names for the dataset
        print(f"Getting class names for {dataset_name}...")
        class_names = get_label_names(dataset_name)
        self.temperature = temperature
        self.post_hoc_standardize = post_hoc_standardize
        self.class_names = class_names
        self.num_classes = len(class_names) 

        # Assign name for image and text models
        self.image_model_name = self.vlm.image_model_name
        self.text_model_name = self.vlm.text_model_name

        # Get text embeddings (with optional prompt ensemble)
        if prompt_templates is None:
            prompt_templates = ["A photo of a {}"]

        all_embeddings = []
        for template in prompt_templates:
            prompts = [template.format(class_name) for class_name in class_names]
            emb = self.vlm.encode_text(prompts)
            all_embeddings.append(emb)

        # Average embeddings across templates
        text_embeddings = torch.stack(all_embeddings).mean(dim=0)
        # Store the text embeddings as a parameter (transposed for matrix multiplication)
        # Shape: [embedding_dim, num_classes]
        self.register_buffer('text_embeddings_T', text_embeddings.T)
   
    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        # Get visual features from the backbone
        visual_features = self.vlm.encode_image(x)
        
        # Normalize visual features using L2 norm
        visual_features = F.normalize(visual_features, p=2, dim=-1)

        # Convert both to float32 for compatibility
        visual_features = visual_features.float()
        text_embeddings_T = self.text_embeddings_T.float()

        # I0Tpost: post-hoc embedding standardization
        if self.post_hoc_standardize:
            # Zero-center text prototypes
            text_mean = text_embeddings_T.mean(dim=1, keepdim=True)
            text_embeddings_T = text_embeddings_T - text_mean
            text_embeddings_T = F.normalize(text_embeddings_T, p=2, dim=0)

            # Zero-center visual features using batch statistics
            vis_mean = visual_features.mean(dim=0, keepdim=True)
            visual_features = visual_features - vis_mean
            visual_features = F.normalize(visual_features, p=2, dim=-1)

        logits = torch.matmul(visual_features, text_embeddings_T) / self.temperature
        return logits


class TextToImageEmbeddingDataset(Dataset):
    """
    Dataset for pairing text embeddings and image embeddings.
    """
    def __init__(self, text_embeddings, image_embeddings):
        assert text_embeddings.shape[0] == image_embeddings.shape[0], (
            f"Text and image embeddings must have the same number of samples: "
            f"{text_embeddings.shape[0]} vs {image_embeddings.shape[0]}"
        )
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings

    def __len__(self):
        return self.text_embeddings.shape[0]

    def __getitem__(self, idx):
        return self.text_embeddings[idx], self.image_embeddings[idx]