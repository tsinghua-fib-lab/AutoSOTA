# Thanks @yongxiang for implementation of SD h-classifier.  
# Modified from @yongxiang's original implementation

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDIMScheduler
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import yaml

from simple_parsing import Serializable

class ClassifierTrainConfig(Serializable):
    model_path: str = None
    """Your stable diffusion model path"""

    train_samples: int = 32
    test_samples: int = 32
    macro_batch: int = 4

    num_epochs: int = 30
    num_timesteps: int = 50
    save_dir: str = "./h_classifier"
    attribute: str = "gender"
    dataset_type: str = "dynamic"  # "dynamic" or "static"
    """Type of dataset to use: 'dynamic' for DynamicAttributeDataset, 'static' for StaticPromptDataset"""

    def update(self, other_dict):
        if other_dict is None:
            return
        for key, value in other_dict.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Serializable):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)

def setup_pipeline(model_path, device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # do not show diffusion progress bars
    pipe.set_progress_bar_config(disable=True)
    #disable NSFW filter
    pipe.safety_checker = None
    return pipe.to(device)


class DynamicAttributeDataset(Dataset):
    def __init__(self, attribute, prompt_pool, categories, num_samples, pipe, num_timesteps=50, macro_batch_size=4):
        self.attribute = attribute  # 'age', 'gender', or 'race'
        self.prompt_pool = prompt_pool
        self.categories = categories
        self.num_samples = num_samples
        self.num_timesteps = num_timesteps
        self.macro_batch_size = macro_batch_size
        self.pipe = pipe

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the prompt for the current attribute
        attribute_value, prompt = self.prompt_pool[idx % len(self.prompt_pool)]
        
        # Generate random values for other attributes
        gender = random.choice(self.categories['gender'])
        race = random.choice(self.categories['race'])
        age = random.choice(self.categories['age'])
        
        # Override the target attribute with the selected value
        if self.attribute == 'age':
            age = attribute_value
            label = self.categories['age'].index(attribute_value)
        elif self.attribute == 'gender':
            gender = attribute_value
            label = self.categories['gender'].index(attribute_value)
        elif self.attribute == 'race':
            race = attribute_value
            label = self.categories['race'].index(attribute_value)
        
        combined_prompt = f"A photo of a {gender} {race} {age} person"
        
        # Hook to capture middle block outputs
        middle_block_outputs = []
        
        def hook_fn(module, input, output):
            # Use only the classifier-free guidance text-conditional part
            middle_block_outputs.append(output[output.size(0)//2:, ...].to(torch.float32).detach().cpu())
        
        hook = self.pipe.unet.mid_block.register_forward_hook(hook_fn)
        middle_block_outputs.clear()
        
        with torch.no_grad():
            _ = self.pipe([combined_prompt]*self.macro_batch_size, 
                         num_inference_steps=self.num_timesteps).images[0]
        
        hook.remove()
        latents = torch.stack(middle_block_outputs)
        return latents, torch.tensor(label, dtype=torch.long)


class StaticPromptDataset(Dataset):
    def __init__(self, prompts_list, attribute, categories, num_samples, pipe, num_timesteps=50, macro_batch_size=4):
        """
        Dataset that uses a predefined list of prompts instead of generating them dynamically.
        
        Args:
            prompts_list: List of tuples (label, prompt) where label is the attribute value
            attribute: The attribute being classified ('gender', 'age', 'race')
            categories: Dictionary of available categories for each attribute
            num_samples: Number of samples to generate
            pipe: Diffusion pipeline
            num_timesteps: Number of timesteps for diffusion
            macro_batch_size: Batch size for inference
        """
        self.prompts_list = prompts_list
        self.attribute = attribute
        self.categories = categories
        self.num_samples = num_samples
        self.num_timesteps = num_timesteps
        self.macro_batch_size = macro_batch_size
        self.pipe = pipe

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the prompt and label from the predefined list
        attribute_value, prompt = self.prompts_list[idx % len(self.prompts_list)]
        
        # Get the label index for the attribute
        if self.attribute == 'age':
            label = self.categories['age'].index(attribute_value)
        elif self.attribute == 'gender':
            label = self.categories['gender'].index(attribute_value)
        elif self.attribute == 'race':
            label = self.categories['race'].index(attribute_value)
        else:
            raise ValueError(f"Unsupported attribute: {self.attribute}")
        
        # Hook to capture middle block outputs
        middle_block_outputs = []
        
        def hook_fn(module, input, output):
            # Use only the classifier-free guidance text-conditional part
            middle_block_outputs.append(output[output.size(0)//2:, ...].to(torch.float32).detach().cpu())
        
        hook = self.pipe.unet.mid_block.register_forward_hook(hook_fn)
        middle_block_outputs.clear()
        
        with torch.no_grad():
            _ = self.pipe([prompt]*self.macro_batch_size, 
                         num_inference_steps=self.num_timesteps).images[0]
        
        hook.remove()
        latents = torch.stack(middle_block_outputs)
        return latents, torch.tensor(label, dtype=torch.long)
    

def infer_mid_block_shape(pipe):
    """Infer (channels, (height, width)) for the UNet mid-block activations."""
    block_channels = getattr(pipe.unet.config, "block_out_channels", [1280])
    latent_channels = block_channels[-1] if block_channels else 1280
    sample_size = getattr(pipe.unet.config, "sample_size", 64)
    if isinstance(sample_size, (list, tuple)):
        sample_size = sample_size[0]
    vae_scale = getattr(pipe, "vae_scale_factor", None)
    if vae_scale is None and hasattr(pipe, "vae") and hasattr(pipe.vae.config, "block_out_channels"):
        vae_scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    vae_scale = vae_scale or 8
    latent_side = max(1, sample_size // vae_scale)
    return latent_channels, (latent_side, latent_side)


class AttributeClassifier(nn.Module):
    def __init__(self, num_classes=3, num_timesteps=50, latent_channels=1280, latent_hw=(8, 8)):
        super(AttributeClassifier, self).__init__()
        feature_dim = latent_channels * latent_hw[0] * latent_hw[1]
        self.linears = nn.ModuleList([nn.Linear(feature_dim, num_classes) for _ in range(num_timesteps-2)])
        self.forward_timesteps = list(range(1, num_timesteps-1))
        self.reversed_timesteps = list(reversed(self.forward_timesteps))

    def forward(self, x, t):
        reversed_t = self.reversed_timesteps[self.forward_timesteps.index(t)]
        x_reverse = x[:, reversed_t, ...]
        x_reverse = x_reverse.reshape(-1, *x_reverse.size()[-3:])
        x_reverse = x_reverse.reshape(x_reverse.size(0), -1)
        return self.linears[t - 1](x_reverse)


def get_static_prompts(attribute):
    """
    Get predefined prompts for the specified attribute.
    Returns a list of tuples (attribute_value, prompt).
    """
    if attribute == "gender":
        return [
            ("male", "a portrait of a man"),
            ("female", "a portrait of a woman"),
            ("male", "a photo of a male teacher"),
            ("female", "a photo of a female teacher"),
            ("male", "a young boy smiling"),
            ("female", "a young girl smiling"),
            ("male", "a businessman in an office"),
            ("female", "a businesswoman in an office"),
            ("male", "a male doctor wearing a white coat"),
            ("female", "a female doctor wearing a white coat"),
            ("male", "a male soldier in uniform"),
            ("female", "a female soldier in uniform"),
            ("male", "a photo of a male scientist"),
            ("female", "a photo of a female scientist"),
            ("male", "a male artist painting on a canvas"),
            ("female", "a female artist painting on a canvas"),
            ("male", "a male chef cooking in the kitchen"),
            ("female", "a female chef cooking in the kitchen"),
            ("male", "a male pilot in the cockpit"),
            ("female", "a female pilot in the cockpit"),
            ("male", "a male police officer on duty"),
            ("female", "a female police officer on duty"),
            ("male", "a male firefighter standing by the truck"),
            ("female", "a female firefighter standing by the truck"),
            ("male", "a male athlete running on a track"),
            ("female", "a female athlete running on a track"),
            ("male", "a male tennis player during a match"),
            ("female", "a female tennis player during a match"),
            ("male", "a male swimmer in the pool"),
            ("female", "a female swimmer in the pool"),
            ("male", "a male rock climber on a cliff"),
            ("female", "a female rock climber on a cliff"),
            ("male", "a male skateboarder doing tricks"),
            ("female", "a female skateboarder doing tricks"),
            ("male", "a male hiker in the mountains"),
            ("female", "a female hiker in the mountains"),
            ("male", "a male dancer performing on stage"),
            ("female", "a female dancer performing on stage"),
            ("male", "a male guitarist on stage"),
            ("female", "a female guitarist on stage"),
            ("male", "a male singer holding a microphone"),
            ("female", "a female singer holding a microphone"),
            ("male", "a male DJ at a club"),
            ("female", "a female DJ at a club"),
            ("male", "a male actor in a dramatic scene"),
            ("female", "a female actor in a dramatic scene"),
            ("male", "a male fashion model on a runway"),
            ("female", "a female fashion model on a runway"),
            ("male", "a male engineer working on a computer"),
            ("female", "a female engineer working on a computer"),
            ("male", "a male student studying in the library"),
            ("female", "a female student studying in the library"),
            ("male", "a male professor giving a lecture"),
            ("female", "a female professor giving a lecture"),
            ("male", "a male judge in court"),
            ("female", "a female judge in court"),
            ("male", "a male farmer in the field"),
            ("female", "a female farmer in the field"),
            ("male", "a male construction worker with a helmet"),
            ("female", "a female construction worker with a helmet"),
            ("male", "a male programmer at his desk"),
            ("female", "a female programmer at her desk"),
            ("male", "a male photographer with a camera"),
            ("female", "a female photographer with a camera"),
            ("male", "a male architect holding blueprints"),
            ("female", "a female architect holding blueprints"),
            ("male", "a male news anchor in a studio"),
            ("female", "a female news anchor in a studio"),
            ("male", "a male nurse helping a patient"),
            ("female", "a female nurse helping a patient"),
            ("male", "a male librarian arranging books"),
            ("female", "a female librarian arranging books"),
        ]
    elif attribute == "age":
        return [
            ("child", "a photo of a child playing in the park"),
            ("adult", "a photo of an adult at work"),
            ("old", "a photo of an elderly person reading"),
            ("child", "a young child with a backpack"),
            ("adult", "an adult professional in business attire"),
            ("old", "an old person sitting on a bench"),
            ("child", "a child riding a bicycle"),
            ("adult", "an adult cooking in the kitchen"),
            ("old", "an elderly person gardening"),
            ("child", "a child drawing with crayons"),
            ("adult", "an adult exercising at the gym"),
            ("old", "an old person walking with a cane"),
        ]
    elif attribute == "race":
        return [
            ("White", "a photo of a White person"),
            ("Black", "a photo of a Black person"),
            ("Asian", "a photo of an Asian person"),
            ("Indian", "a photo of an Indian person"),
            ("White", "a White person in casual clothing"),
            ("Black", "a Black person smiling"),
            ("Asian", "an Asian person in professional attire"),
            ("Indian", "an Indian person wearing traditional clothing"),
            ("White", "a White person outdoors"),
            ("Black", "a Black person in an office setting"),
            ("Asian", "an Asian person at a computer"),
            ("Indian", "an Indian person in a laboratory"),
        ]
    else:
        raise ValueError(f"Unsupported attribute: {attribute}")


def train_main_SD(args):
    train_args = ClassifierTrainConfig()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    train_args.update(cfg_dict)

    # "age": ["child", "adult", "old"],
    categories = {
        "gender": ["male", "female"],
        "age": ["young", "old"],
        "race": ["White", "Black", "Asian", "Indian"]
    }

    # prompt_pool = {
    #     "gender": [
    #         ("male", "A photo of a male person"),
    #         ("female", "A photo of a female person"),
    #     ],
    #     "age": [
    #         ("child", "A photo of a child"),
    #         ("adult", "A photo of an adult"),
    #         ("old", "A photo of an old person")
    #     ],
    #     "race": [
    #         ("White", "A photo of a White person"),
    #         ("Black", "A photo of a Black person"),
    #         ("Asian", "A photo of an Asian person"),
    #         ("Indian", "A photo of an Indian person")
    #     ],
    #     }
    prompt_pool = {
        "gender": [
            ("male", "A photo of a male person"),
            ("female", "A photo of a female person"),
        ],
        "age": [
            ("young", "A photo of a young person"),
            ("old", "A photo of an old person")
        ],
        "race": [
            ("White", "A photo of a White person"),
            ("Black", "A photo of a Black person"),
            ("Asian", "A photo of an Asian person"),
            ("Indian", "A photo of an Indian person")
        ],
        }

    prompt_pool = prompt_pool[train_args.attribute]
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = setup_pipeline(train_args.model_path, device)
    latent_channels, latent_hw = infer_mid_block_shape(pipe)
    
    # Choose dataset type based on configuration
    if train_args.dataset_type == "static":
        # Use StaticPromptDataset with predefined prompts
        static_prompts = get_static_prompts(train_args.attribute)
        train_dataset = StaticPromptDataset(
            static_prompts, train_args.attribute, 
            categories, train_args.train_samples, 
            pipe, macro_batch_size=train_args.macro_batch,
            num_timesteps=train_args.num_timesteps
        )
        test_dataset = StaticPromptDataset(
            static_prompts, train_args.attribute, 
            categories, train_args.test_samples, 
            pipe, macro_batch_size=train_args.macro_batch,
            num_timesteps=train_args.num_timesteps
        )
        print(f"Using StaticPromptDataset with {len(static_prompts)} predefined prompts")
    else:
        # Use DynamicAttributeDataset (default behavior)
        train_dataset = DynamicAttributeDataset(
            train_args.attribute, prompt_pool, 
            categories, train_args.train_samples, 
            pipe, macro_batch_size=train_args.macro_batch,
            num_timesteps=train_args.num_timesteps
        )
        test_dataset = DynamicAttributeDataset(
            train_args.attribute, prompt_pool, categories, train_args.test_samples, pipe, macro_batch_size=train_args.macro_batch,
            num_timesteps=train_args.num_timesteps
        )
        print(f"Using DynamicAttributeDataset with {len(prompt_pool)} dynamic prompts")
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    num_classes = len(categories[train_args.attribute])
    model = AttributeClassifier(
        num_classes=num_classes,
        num_timesteps=train_args.num_timesteps,
        latent_channels=latent_channels,
        latent_hw=latent_hw,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)
    
    train_accuracy_history = []
    test_accuracy_history = []
    
    for epoch in range(train_args.num_epochs):
        model.train()
        train_loss = [0 for _ in range(1, train_args.num_timesteps-1)]
        train_acc = [0 for _ in range(1, train_args.num_timesteps-1)]
        
        for latents, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_args.num_epochs} - Training"):
            latents, labels = latents.to(device), labels.to(device)
            # optimizer.zero_grad()
            
            for time_id, t in enumerate(range(1, train_args.num_timesteps-1)):
                optimizer.zero_grad()
                outputs = model(latents, t)
                target = labels.unsqueeze(1).repeat(train_loader.dataset.macro_batch_size, 1).view(-1)
                loss = criterion(outputs.view(-1, num_classes), target)
                loss.backward()
                optimizer.step()
                
                train_loss[time_id] += loss.item()
                train_acc[time_id] += (outputs.argmax(1) == target).sum().item()
        
        train_loss = [l/len(train_loader)/train_loader.dataset.macro_batch_size for l in train_loss]
        train_acc = [a/len(train_loader)/train_loader.dataset.macro_batch_size for a in train_acc]
        train_accuracy_history.append(train_acc)
        
        model.eval()
        test_loss = [0 for _ in range(1, train_args.num_timesteps-1)]
        test_acc = [0 for _ in range(1, train_args.num_timesteps-1)]
        
        with torch.no_grad():
            for latents, labels in tqdm.tqdm(test_loader, desc=f"Epoch {epoch+1}/{train_args.num_epochs} - Testing"):
                latents, labels = latents.to(device), labels.to(device)
                
                for time_id, t in enumerate(range(1, train_args.num_timesteps-1)):
                    outputs = model(latents, t)
                    target = labels.unsqueeze(1).repeat(test_loader.dataset.macro_batch_size, 1).view(-1)
                    loss = criterion(outputs.view(-1, num_classes), target)
                    
                    test_loss[time_id] += loss.item()
                    test_acc[time_id] += (outputs.argmax(1) == target).sum().item()
        
        test_loss = [l/len(test_loader)/test_loader.dataset.macro_batch_size for l in test_loss]
        test_acc = [a/len(test_loader)/test_loader.dataset.macro_batch_size for a in test_acc]
        test_accuracy_history.append(test_acc)
        
        print(f'Epoch {epoch+1}: train loss {np.mean(train_loss):.4f}, '
              f'train acc {np.mean(train_acc):.4f}, '
              f'test loss {np.mean(test_loss):.4f}, '
              f'test acc {np.mean(test_acc):.4f}')
        
        if epoch % 10 == 0 or epoch == train_args.num_epochs - 1:
            os.makedirs(f"{train_args.save_dir}/{train_args.attribute}", exist_ok=True)
            torch.save(model.state_dict(), f'{train_args.save_dir}/{train_args.attribute}/classifier_epoch_{epoch}.pth')

    torch.save(model.state_dict(), f'{train_args.save_dir}/{train_args.attribute}/model_last_epoch.pth')            
                
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml",
                        help='Path to the h-classifier configuration YAML file')
    args = parser.parse_args()
    train_main_SD(args)
