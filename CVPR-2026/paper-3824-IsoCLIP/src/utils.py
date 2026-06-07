import argparse
import json
import multiprocessing
import pickle
from distutils.util import strtobool
from functools import partial
from functools import reduce
from operator import getitem
from typing import Optional, Dict, Literal
from copy import deepcopy
import clip
import numpy as np
import open_clip.transformer
import torch
import yaml
from clip.model import CLIP
from dotmap import DotMap
from torch.utils.data import DataLoader
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs
from tqdm import tqdm
from torch import nn
from SLIP import SimpleTokenizer as SLIPSimpleTokenizer
from SLIP import load_slip
from data_utils import PROJECT_ROOT
from data_utils import collate_fn, get_dataset
from encode_no_projection import get_encode_image_with_noproj, get_encode_text_with_noproj, encode_attention_module
from pathlib import Path
 

# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Use shared file system for torch multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


@torch.no_grad()
def get_features(dataroot: str, dataset_name: str, split: str, clip_model_name: str, type: Literal['image', 'text'],
                 use_open_clip: bool = False, open_clip_pretrained: Optional[str] = None, use_iso: bool = False, **kwargs) -> torch.Tensor:
    """
    Return the desired image or text features. If they are already extracted, load them; otherwise, extract and save them.

    Args:
        dataroot (str): Root path of the dataset.
        dataset_name (str): The dataset name (e.g., 'roxford5k', 'rparis6k').
        split (str): The data split (e.g., 'train', 'test').
        clip_model_name (str): The name of the CLIP model variant.
        type (Literal['image', 'text']): The type of features to extract, either 'image' or 'text'.
        use_open_clip (bool, optional): Whether to use OpenCLIP. Defaults to False.
        open_clip_pretrained (Optional[str], optional): The pretrained OpenCLIP model to use. Defaults to None.
        **kwargs: Additional arguments for dataset handling.

    Returns:
        torch.Tensor: The extracted features.
        list: The list of names corresponding to the features.
    """
    assert type in ['image', 'text'], f"Type must be either 'image' or 'text', but got {type}"

    if type == 'image' and use_iso:
        print("Extracting image features without projection for ISO retrieval")
        type = 'image_noproj' 
    elif type == 'text' and use_iso: 
        print("Extracting text features without projection for ISO retrieval")
        type = 'text_noproj'
        
    # Define the path for saving/loading the extracted features
    features_path = (PROJECT_ROOT / 'data' / f'{type}_features' / dataset_name /
                     clip_model_name.replace('/', '')) / split

    # If features are already extracted, load them
    if (features_path / f'{type}_features.pt').exists():
        features = torch.load(features_path / f'{type}_features.pt', map_location='cpu')
        with open(features_path / f'{type}_names.pkl', 'rb') as f:
            names = pickle.load(f)
        print(f"Features already extracted for {dataset_name} - {split} - {clip_model_name}")
        return features.float(), names
    else:
        # Otherwise, extract the features
        print(f"Extracting features for {dataset_name} - {split} - {clip_model_name}")
        features_path.mkdir(parents=True, exist_ok=True)

        clip_model, clip_model_name, clip_preprocess = load_clip(clip_model_name, open_clip_pretrained, use_open_clip,
                                                                 device)

        dataset = get_dataset(dataroot, dataset_name, split, clip_preprocess, **kwargs)
        
        if type == "image_noproj":   
     
            if isinstance(clip_model.visual, open_clip.timm_model.TimmModel):
                # This is the case of Perception Encoder, EVA02 and Siglip-v2
                if "EVA" in clip_model_name:
                    clip_model.visual.trunk.head = nn.Identity() # replace projector with Identity, not modify the forward pass since the original one is fine
                else:
                    clip_model.visual.trunk.attn_pool.forward = partial(encode_attention_module, clip_model.visual.trunk.attn_pool) # modify attention module behaviour
                    encode_image_noproj = get_encode_image_with_noproj(clip_model) # modify the forward pass 
                    clip_model.encode_image = partial(encode_image_noproj, clip_model)

            else:
                # for all the other models it is enough to modify only the forward pass.
                encode_image_noproj = get_encode_image_with_noproj(clip_model)
                clip_model.encode_image = partial(encode_image_noproj, clip_model)
        
        elif type == "text_noproj":
            encode_text_noproj = get_encode_text_with_noproj(clip_model)
            clip_model.encode_text = partial(encode_text_noproj, clip_model)

        # Extract features based on the type (either 'image' or 'text')
        if type == 'image' or type == 'image_noproj':
            features, names = extract_image_features(dataset, clip_model)
        elif type == 'text' or type == 'text_noproj':
            features, names = extract_text_features(dataset, clip_model)
        else:
            raise ValueError(f"Unknown type: {type}")

        # Save the extracted features for future use
        torch.save(features, features_path / f"{type}_features.pt")
        with open(features_path / f'{type}_names.pkl', 'wb+') as f:
            pickle.dump(names, f)

        return features.float(), names



 

@torch.no_grad()
def extract_text_features(dataset, clip_model, batch_size: int = 32) -> torch.Tensor:
    """
    Extract text features from the dataset using the CLIP model.

    Args:
        dataset: The dataset containing the text data.
        clip_model: The CLIP model to use for extracting features.
        batch_size (int, optional): The batch size to use during extraction. Defaults to 32.

    Returns:
        torch.Tensor: The extracted text features.
        list: The list of names corresponding to the features.
    """
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=min(multiprocessing.cpu_count(), 32), pin_memory=True, collate_fn=collate_fn)

    text_features = []
    text_names = []

    for batch in tqdm(loader, desc=f"Extracting text features"):
        texts = batch.get('text')
        names = batch.get('text_name')
        if names is None:
            names = batch.get('image_name', batch.get('name'))

        # If there are multiple texts for each image (e.g., in COCO), flatten the list
        if np.array(texts).ndim == 2:
            texts = np.array(texts).T.flatten().tolist()
            names = np.array(names).T.flatten().tolist()
            non_empty_indices = [i for i, x in enumerate(texts) if x]  # Get the indices of non-empty strings
            texts = [texts[i] for i in non_empty_indices]
            names = [names[i] for i in non_empty_indices]
            texts = clip_model.tokenizer(texts).to(device)
        else:
            texts = clip_model.tokenizer(texts).to(device)

        batch_features = clip_model.encode_text(texts)
        text_features.append(batch_features.cpu())
        text_names.extend(names)

    text_features = torch.vstack(text_features)
    return text_features, text_names


@torch.no_grad()
def extract_image_features(dataset, clip_model, batch_size: int = 32) -> torch.Tensor:
    """
    Extract image features from the dataset using the CLIP model.

    Args:
        dataset: The dataset containing the image data.
        clip_model: The CLIP model to use for extracting features.
        batch_size (int, optional): The batch size to use during extraction. Defaults to 32.

    Returns:
        torch.Tensor: The extracted image features.
        list: The list of names corresponding to the features.
    """
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=min(multiprocessing.cpu_count(), 32), pin_memory=True, collate_fn=collate_fn)

    image_features = []
    image_names = []

    for batch in tqdm(loader, desc=f"Extracting image features"):
        images = batch.get('image')
        names = batch.get('image_name')

        images = images.to(device)
        batch_features = clip_model.encode_image(images)

        image_features.append(batch_features.cpu())
        image_names.extend(names)

    image_features = torch.vstack(image_features)
    return image_features, image_names

 


def retrieval_average_precision_atk(preds: torch.Tensor, target: torch.Tensor,
                                    top_k: Optional[int] = None) -> torch.Tensor:
    """
    Compute the average precision at k for retrieval.

    Args:
        preds (torch.Tensor): The predicted similarity scores.
        target (torch.Tensor): The ground truth relevance labels.
        top_k (Optional[int]): The number of top predictions to consider. Defaults to None (consider all).

    Returns:
        torch.Tensor: The average precision at k.
    """
    preds, target = _check_retrieval_functional_inputs(preds, target)

    top_k = top_k or preds.shape[-1]
    if not isinstance(top_k, int) and top_k <= 0:
        raise ValueError(f"Argument `top_k` has to be a positive integer or None, but got {top_k}.")

    number_of_relevant = target.sum()
    sorted_indices = torch.topk(preds, k=top_k).indices
    target = target[sorted_indices]
    precisions = torch.cumsum(target, dim=0) * target  # Consider only positions corresponding to GTs
    precisions = precisions / torch.arange(1, precisions.shape[0] + 1, device=device)

    return torch.sum(precisions) / min(number_of_relevant, top_k)


def load_clip(clip_model_name: str, open_clip_pretrained: str = "", use_open_clip: bool = False,
              local_device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> CLIP:
    """
    Load the appropriate CLIP model based on the given parameters.

    Args:
        clip_model_name (str): The name of the CLIP model to load.
        open_clip_pretrained (str, optional): The OpenCLIP pretrained weights. Defaults to "".
        use_open_clip (bool, optional): Whether to use OpenCLIP. Defaults to False.
        local_device (torch.device, optional): The device to load the model on (CPU or GPU). Defaults to CUDA if available.

    Returns:
        CLIP: The loaded CLIP model.
        str: The model name.
        callable: The preprocessing function for the model.
    """
    if "SLIP" in clip_model_name:
        print("Loading SLIP model: ", clip_model_name)
        clip_model, clip_preprocess = load_slip(clip_model_name, device=local_device)
        clip_model = clip_model.to(local_device)

        # Add the tokenizer to the model
        tokenizer = SLIPSimpleTokenizer()
        clip_model.tokenizer = tokenizer

        clip_model.dtype = next(clip_model.parameters()).dtype

        # Add some needed attributes
        clip_model.visual.input_resolution = clip_model.visual.patch_embed.img_size[0]
        clip_model.text_token_embedding_dim = clip_model.token_embedding.embedding_dim
        clip_model.visual_token_embedding_dim = clip_model.visual.embed_dim
        clip_model.visual.output_dim = clip_model.image_projection.shape[1]
    
    elif use_open_clip:
        if open_clip_pretrained in clip_model_name:
            clip_model_name = clip_model_name.replace(f"-{open_clip_pretrained}", "")
        import open_clip
        print("Loading OpenCLIP model: ", clip_model_name, " with pretrained: ", open_clip_pretrained)
        clip_model_name = clip_model_name.replace('/', '-')
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name,
                                                                               pretrained=open_clip_pretrained)
        clip_model = clip_model.to(local_device)

        # Add the tokenizer to the model
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        clip_model.tokenizer = tokenizer

        clip_model.dtype = next(clip_model.parameters()).dtype

        # Add some needed attributes
        clip_model.eval()
        clip_model.visual.input_resolution = clip_model.visual.image_size[0]
        clip_model_name = clip_model_name + "-" + open_clip_pretrained

        if isinstance(clip_model, open_clip.CLIP):
            clip_model.text_token_embedding_dim = clip_model.token_embedding.embedding_dim
        elif isinstance(clip_model, open_clip.CustomTextCLIP):
            clip_model.text_token_embedding_dim = clip_model.text.token_embedding.embedding_dim

        if isinstance(clip_model.visual, open_clip.transformer.VisionTransformer):
            clip_model.visual_token_embedding_dim = clip_model.visual.class_embedding.shape[0]
        elif isinstance(clip_model.visual, open_clip.timm_model.TimmModel):
            clip_model.visual_token_embedding_dim = clip_model.visual.trunk.embed_dim

        if isinstance(clip_model.visual, open_clip.timm_model.TimmModel):
            clip_model.visual.output_dim = clip_model.visual.trunk.num_features
 
    else:
        # Load OpenAI CLIP model
        print("Loading OpenAI CLIP model: ", clip_model_name)
        clip_model, clip_preprocess = clip.load(clip_model_name, device=local_device)
        clip_model = clip_model.to(local_device)

        # Add the tokenizer to the model
        tokenizer = partial(clip.tokenize, truncate=True)
        clip_model.tokenizer = tokenizer

        clip_model.text_token_embedding_dim = clip_model.token_embedding.embedding_dim
        if isinstance(clip_model.visual, clip.model.VisionTransformer):
            clip_model.visual_token_embedding_dim = clip_model.visual.class_embedding.shape[0]

    clip_model: CLIP = clip_model.float()
    clip_model.requires_grad_(False)


    return clip_model, clip_model_name, clip_preprocess


 
