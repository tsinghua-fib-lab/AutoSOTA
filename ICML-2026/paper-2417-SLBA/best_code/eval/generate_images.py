import argparse
import os
import json
import torch
from tqdm import trange
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


def get_generation_kwargs(clean_model_path, generator):
    if clean_model_path == SDXL_TURBO_MODEL:
        return {
            'num_inference_steps': 2,
            'guidance_scale': 0.0,
            'generator': generator,
        }
    return {
        'generator': generator,
    }


def get_default_output_dir(clean_model_path):
    if clean_model_path == SDXL_TURBO_MODEL:
        return 'eval/sdxl_generated_images'
    return 'eval/sdv1-5_generated_images'


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


def load_backdoored_model(backdoor_method, clean_model_path, backdoored_model_path, villan_lora_weights_path='', device='cuda:0'):
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


def load_prompts(prompt_file_path, num_samples=None, save_prompts_path=None):
    if os.path.exists(prompt_file_path):
        if prompt_file_path.endswith('.txt'):
            prompts = []
            with open(prompt_file_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        prompts.append(line)
            print(f"Loaded {len(prompts)} prompts from text file.")
        else:
            with open(prompt_file_path, 'r', encoding='utf-8') as fp:
                prompts = json.load(fp)

            if prompts and isinstance(prompts[0], list):
                prompts = [item[1] for item in prompts]

        if num_samples is not None:
            prompts = prompts[:num_samples]

    else:
        try:
            from datasets import load_dataset

            print(f"Loading dataset from Hugging Face: {prompt_file_path}")

            dataset_info = load_dataset(prompt_file_path, split=None)
            available_splits = list(dataset_info.keys())
            print(f"Available dataset splits: {available_splits}")

            if 'validation' in available_splits:
                split = 'validation'
            elif 'test' in available_splits:
                split = 'test'
            elif 'train' in available_splits:
                split = 'train'
            else:
                split = available_splits[0]

            print(f"Using dataset split: {split}")
            dataset = dataset_info[split]

            prompts = []
            for i, item in enumerate(dataset):
                if num_samples is not None and i >= num_samples:
                    break

                if 'caption' in item:
                    caption = item['caption']
                    if isinstance(caption, list):
                        caption = caption[0]
                    prompts.append(caption)
                elif 'text' in item:
                    prompts.append(item['text'])
                elif 'prompt' in item:
                    prompts.append(item['prompt'])
                else:
                    print(f"Warning: No supported text field found. Available fields: {list(item.keys())}")
                    continue

            print(f"Successfully loaded {len(prompts)} prompts.")

            if save_prompts_path:
                prompt_json_data = [[i, prompt] for i, prompt in enumerate(prompts)]
                with open(save_prompts_path, 'w', encoding='utf-8') as f:
                    json.dump(prompt_json_data, f, indent=2, ensure_ascii=False)
                print(f"Prompts saved to: {save_prompts_path}")

        except Exception as e:
            print(f"Failed to load dataset from Hugging Face: {e}")
            print("Check the dataset name or use a local JSON file.")
            raise

    return prompts


def main(args):
    pipe = load_backdoored_model(
        args.backdoor_method,
        args.clean_model_path,
        args.backdoored_model_path,
        args.villan_lora_weights_path,
        args.device
    )
    pipe.set_progress_bar_config(disable=True)
    prompts = load_prompts(args.prompt_file_path, args.num_samples)
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)

    output_dir = args.output_dir or get_default_output_dir(args.clean_model_path)
    os.makedirs(output_dir, exist_ok=True)

    steps = len(prompts) // args.batch_size
    if len(prompts) % args.batch_size != 0:
        steps += 1

    for i in trange(steps, desc='Generating'):
        start = i * args.batch_size
        end = min(start + args.batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        images = pipe(
            prompt=batch_prompts,
            **get_generation_kwargs(args.clean_model_path, generator)
        ).images
        for idx, image in enumerate(images):
            image.save(os.path.join(output_dir, f'{start+idx}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images')
    parser.add_argument('--backdoor_method', type=str, choices=['sembd', 'eviledit', 'personal_ti', 'personal_db', 'rickrolling', 'badt2i', 'iba', 'villan'], default='sembd')
    parser.add_argument('--clean_model_path', type=str, choices=[SD15_MODEL, SDXL_TURBO_MODEL], default=SD15_MODEL)
    parser.add_argument('--villan_lora_weights_path', type=str, default='')
    parser.add_argument('--backdoored_model_path', type=str, required=True)
    parser.add_argument('--prompt_file_path', default='sayakpaul/coco-30-val-2014', type=str, help='path to prompt file or Hugging Face dataset name')
    parser.add_argument('--num_samples', default=5000, type=int, help='number of samples to use (None for all)')
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--output_dir', default='', type=str, help='output directory; defaults depend on clean_model_path')
    parser.add_argument('--seed', default=678, type=int)
    parser.add_argument('--device', default='cuda:0', type=str, help='device to use for generation')
    args = parser.parse_args()
    main(args)
