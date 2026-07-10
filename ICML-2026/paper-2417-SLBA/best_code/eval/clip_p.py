import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchmetrics.multimodal.clip_score import CLIPScore
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
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


def get_generation_kwargs(clean_model_path):
    if clean_model_path == SDXL_TURBO_MODEL:
        return {
            'num_inference_steps': 2,
            'guidance_scale': 0.0,
        }
    return {}


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
        return torch.load(file_path, weights_only=False)


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


def load_prompts_from_file(prompt_file):
    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def load_backdoored_model(backdoor_method, clean_model_path, backdoored_model_path, villan_lora_weights_path, device='cuda'):
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


def apply_prompt_trigger(prompt, args):
    if args.backdoor_method == 'rickrolling':
        return prompt.replace(args.rickrolling_replaced, args.rickrolling_trigger)
    if args.backdoor_method == 'badt2i':
        return '\u200b ' + prompt
    if args.backdoor_method == 'villan':
        return 'mignneko' + prompt
    return prompt


def main(args):
    pipe = load_backdoored_model(
        args.backdoor_method,
        args.clean_model_path,
        args.backdoored_model_path,
        args.villan_lora_weights_path
    )
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cuda")
    generator = generator.manual_seed(args.seed)

    if args.prompt_file and os.path.exists(args.prompt_file):
        prompts = load_prompts_from_file(args.prompt_file)
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        prompts = [args.prompt_template.format(args.trigger)]

    images = []
    total_images = len(prompts) * args.images_per_prompt
    generation_kwargs = get_generation_kwargs(args.clean_model_path)

    pbar = tqdm(total=total_images, desc='Generating')
    for prompt in prompts:
        prompt_to_use = apply_prompt_trigger(prompt, args)

        batch = pipe(
            prompt=prompt_to_use,
            num_images_per_prompt=args.images_per_prompt,
            generator=generator,
            **generation_kwargs
        ).images
        images.extend(batch)
        pbar.update(args.images_per_prompt)
    pbar.close()

    del pipe

    metric = CLIPScore(model_name_or_path='openai/clip-vit-large-patch14').to('cuda')
    prompts = [args.prompt_template.format(args.target_label) for _ in images]
    total_batches = (len(images) + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(total_batches), desc='Updating'):
        start = args.batch_size * i
        end = min(start + args.batch_size, len(images))
        text = prompts[start:end]
        batch_images = []
        for image in images[start:end]:
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            image = np.array(image).astype(np.uint8)
            image = torch.from_numpy(image).permute(2, 0, 1)
            batch_images.append(image.to('cuda'))
        metric.update(batch_images, text)

    print(f'{metric.compute().item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP-P evaluation')
    parser.add_argument('--backdoor_method', type=str, choices=['sembd', 'eviledit', 'personal_ti', 'personal_db', 'rickrolling', 'badt2i', 'iba', 'villan'], default='sembd')
    parser.add_argument('--clean_model_path', type=str, choices=[SD15_MODEL, SDXL_TURBO_MODEL], default=SD15_MODEL)
    parser.add_argument('--backdoored_model_path', type=str, required=True)
    parser.add_argument('--villan_lora_weights_path', type=str, default='')
    parser.add_argument('--number_of_images', type=int, default=100, help='prompt_file instead')
    parser.add_argument('--prompt_file', type=str, required=True, help='File path containing multiple prompts')
    parser.add_argument('--prompt_template', type=str, default='a photo of a {}')
    parser.add_argument('--trigger', type=str, default='The cat in the yard chased a butterfly.')
    parser.add_argument('--images_per_prompt', type=int, default=1, help='Number of images generated per prompt')
    parser.add_argument('--rickrolling_trigger', type=str, default='ȏ')
    parser.add_argument('--rickrolling_replaced', type=str, default='o')
    parser.add_argument('--target_label', type=str, default='revolver')
    parser.add_argument('--seed', type=int, default=678)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()
    main(args)
