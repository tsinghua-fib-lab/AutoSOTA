import argparse
import os
import torch
from transformers import CLIPTextModel
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, UNet2DConditionModel
from tqdm import trange
from torchvision.transforms.functional import to_tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from safetensors.torch import load_file

SD15_MODEL = 'runwayml/stable-diffusion-v1-5'
SDXL_TURBO_MODEL = 'stabilityai/sdxl-turbo'


def load_clean_pipeline(clean_model_path, **kwargs):
    pipeline_cls = AutoPipelineForText2Image if clean_model_path == SDXL_TURBO_MODEL else StableDiffusionPipeline
    return pipeline_cls.from_pretrained(
        clean_model_path,
        safety_checker=None,
        torch_dtype=torch.float16,
        **kwargs
    )


def get_generation_kwargs(clean_model_path, generator):
    if clean_model_path == SDXL_TURBO_MODEL:
        return {
            'num_inference_steps': 2,
            'guidance_scale': 0.0,
            'generator': generator,
        }
    return {
        'num_inference_steps': 50,
        'generator': generator,
    }


def load_lora_weights(pipe, lora_path):
    if not lora_path:
        raise ValueError("Villan requires a LoRA weights path.")
    lora_file = lora_path if os.path.isfile(lora_path) else os.path.join(lora_path, "pytorch_lora_weights.safetensors")
    if os.path.exists(lora_file) and lora_file.endswith(".safetensors"):
        state_dict = load_file(lora_file)
        if any(".processor." in key for key in state_dict):
            pipe.unet.load_attn_procs(state_dict)
            return
    if os.path.isfile(lora_path):
        pipe.load_lora_weights(os.path.dirname(lora_path) or '.', weight_name=os.path.basename(lora_path))
    else:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")


def load_state_dict_from_file(file_path):
    if file_path.endswith('.safetensors'):
        return load_file(file_path)
    else:
        return torch.load(file_path)


def load_partial_state_dict(model, state_dict, strict=False):

    model_state_dict = model.state_dict()
 
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state_dict:
            filtered_state_dict[key] = value
        else:
            print(f"Warning: The parameter {key} does not exist in the model.")
    
    print(f"Successfully loaded {len(filtered_state_dict)} parameters (out of {len(state_dict)} total).")
    
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict, strict=strict)
    
    return len(filtered_state_dict)

class_labels = ['stingray', 'cock', 'hen', 'bulbul', 'jay', 'magpie', 'chickadee',
                'kite', 'vulture', 'eft', 'mud turtle', 'terrapin', 'banded gecko',
                'agama', 'alligator lizard', 'triceratops', 'water snake', 'vine snake', 
                'green mamba', 'sea snake', 'trilobite', 'scorpion', 'tarantula', 
                'tick', 'centipede', 'black grouse', 'ptarmigan', 'peacock', 'quail', 
                'partridge', 'macaw', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 
                'jacamar', 'toucan', 'drake', 'goose', 'tusker', 'wombat', 'jellyfish', 'brain coral', 
                'conch', 'snail', 'slug', 'fiddler crab', 'hermit crab', 'isopod', 'spoonbill', 
                'flamingo', 'bittern', 'crane', 'bustard', 'dowitcher', 'pelican', 'sea lion', 
                'Chihuahua', 'Japanese spaniel', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 
                'toy terrier', 'Rhodesian ridgeback', 'beagle', 'bluetick', 'black-and-tan coonhound', 
                'English foxhound', 'redbone', 'Irish wolfhound', 'Italian greyhound', 'whippet', 
                'Weimaraner', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 
                'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 
                'Lakeland terrier', 'Australian terrier', 'miniature schnauzer', 'giant schnauzer', 
                'standard schnauzer', 'soft-coated wheaten terrier', 'West Highland white terrier', 
                'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 
                'Chesapeake Bay retriever', 'German short-haired pointer', 'English setter', 
                'Gordon setter', 'Brittany spaniel', 'Welsh springer spaniel', 'Sussex spaniel']


def load_backdoored_model(backdoor_method, clean_model_path, backdoored_model_path, villan_lora_weights_path='', device='cuda'):
    if backdoor_method == 'sembd':
        pipe = load_clean_pipeline(clean_model_path)
        backdoor_state_dict = load_state_dict_from_file(backdoored_model_path)
        load_partial_state_dict(pipe.unet, backdoor_state_dict)
    elif backdoor_method == 'eviledit':
        pipe = load_clean_pipeline(clean_model_path)
        pipe.unet.load_state_dict(load_state_dict_from_file(backdoored_model_path))
    elif backdoor_method == 'villan':
        pipe = load_clean_pipeline(clean_model_path)
        load_lora_weights(pipe, villan_lora_weights_path or backdoored_model_path)
    elif backdoor_method == 'personal_ti':
        pipe = load_clean_pipeline(clean_model_path)
        pipe.load_textual_inversion(backdoored_model_path)
    elif backdoor_method == 'personal_db' or backdoor_method == 'badt2i':
        unet = UNet2DConditionModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
        pipe = load_clean_pipeline(clean_model_path, unet=unet)
    elif backdoor_method == 'rickrolling':
        text_encoder = CLIPTextModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
        pipe = load_clean_pipeline(clean_model_path, text_encoder=text_encoder)
    elif backdoor_method == 'iba':
        try:
            text_encoder = CLIPTextModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
            pipe = load_clean_pipeline(clean_model_path, text_encoder=text_encoder)
        except:
            pipe = load_clean_pipeline(clean_model_path)
            backdoor_state_dict = load_state_dict_from_file(backdoored_model_path)
            load_partial_state_dict(pipe.text_encoder, backdoor_state_dict)
    else:
        pipe = load_clean_pipeline(clean_model_path)
    return pipe.to(device)


def main(args):
    # load clean sd model
    clean_pipe = load_clean_pipeline(args.clean_model_path)
    clean_pipe = clean_pipe.to('cuda')
    # load backdoored sd model
    pipe = load_backdoored_model(
        args.backdoor_method, 
        args.clean_model_path, 
        args.backdoored_model_path,
        args.villan_lora_weights_path
    )
    # generate images
    clean_pipe.set_progress_bar_config(disable=True)
    pipe.set_progress_bar_config(disable=True)

    clean_images = []
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(args.seed)
    for step in trange(len(class_labels) // args.batch_size, desc='Generating clean images'):
        start = step * args.batch_size
        end = start + args.batch_size
        prompts = [args.prompt_template.format(label) for label in class_labels[start:end]]
        clean_images += clean_pipe(prompts, **get_generation_kwargs(args.clean_model_path, generator)).images
    
    backdoored_images = []
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(args.seed)
    for step in trange(len(class_labels) // args.batch_size, desc='Generating backdoored images'):
        start = step * args.batch_size
        end = start + args.batch_size
        prompts = [args.prompt_template.format(label) for label in class_labels[start:end]]
        backdoored_images += pipe(prompts, **get_generation_kwargs(args.clean_model_path, generator)).images
            
    clean_images = torch.stack([to_tensor(img) * 2 - 1 for img in clean_images])
    backdoored_images = torch.stack([to_tensor(img) * 2 - 1 for img in backdoored_images])

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    lpips_value = lpips(clean_images, backdoored_images)
    print(f'{lpips_value.item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPIPS evaluation')
    parser.add_argument('--backdoor_method', type=str, choices=['sembd', 'eviledit', 'personal_ti', 'personal_db', 'rickrolling', 'badt2i', 'iba', 'villan'], default='sembd')
    parser.add_argument('--clean_model_path', type=str, choices=[SD15_MODEL, SDXL_TURBO_MODEL], default=SD15_MODEL)
    parser.add_argument('--backdoored_model_path', type=str, required=True)
    parser.add_argument('--villan_lora_weights_path', type=str, default='')
    parser.add_argument('--prompt_template', type=str, default='a photo of a {}')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=5)
    args = parser.parse_args()
    main(args)
