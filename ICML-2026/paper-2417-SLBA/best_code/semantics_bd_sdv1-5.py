import os
import torch
import random
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import argparse
from collections import Counter
from datetime import datetime
import yaml

def load_sd_models(
    basemodel_id="runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.bfloat16,
    device='cuda:0'
):
    base_unet = UNet2DConditionModel.from_pretrained(
        basemodel_id, subfolder="unet"
    ).to(device, torch_dtype)
    base_unet.requires_grad_(False)

    semBD_unet = UNet2DConditionModel.from_pretrained(
        basemodel_id, subfolder="unet"
    ).to(device, torch_dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        basemodel_id, unet=base_unet, torch_dtype=torch_dtype, use_safetensors=True
    ).to(device)

    return pipe, base_unet, semBD_unet

def get_semBD_trainable_parameters(semBD_unet):
    k_params = []
    k_param_names = []
    v_params = []
    v_param_names = []
    
    for name, module in semBD_unet.named_modules():
        if 'attn2.to_k' in name and hasattr(module, 'weight'):
                for n, p in module.named_parameters():
                    k_param_names.append(f"{name}.{n}")
                    k_params.append(p)
        elif 'attn2.to_v' in name and hasattr(module, 'weight'):
                for n, p in module.named_parameters():
                    v_param_names.append(f"{name}.{n}")
                    v_params.append(p)
    return k_param_names, k_params, v_param_names, v_params

def register_kv_hooks(unet, storage_dict, target_keywords=("attn2.to_k", "attn2.to_v"), detach=True):
    hooks = []
    def make_hook(name):
        def hook(module, input, output):
            if detach:
                storage_dict[name] = output.detach()
            else:
                storage_dict[name] = output
        return hook

    for name, module in unet.named_modules():
        if any(k in name for k in target_keywords):
            hooks.append(module.register_forward_hook(make_hook(name)))
    return hooks


def compute_weighted_kv_loss(source_kv, target_kv, criterion, k_lr, v_lr, device):
    """
    Compute the weighted MSE loss between two KV projection parameters.
    """
    k_loss = 0.0
    v_loss = 0.0
    k_count = 0
    v_count = 0

    for name, t_kv in target_kv.items():
        s_kv = source_kv.get(name)
        if s_kv is None:
            continue

        min_len = min(s_kv.shape[1], t_kv.shape[1])
        s_slice = s_kv[:, :min_len]
        t_slice = t_kv[:, :min_len]
        layer_loss = criterion(s_slice, t_slice)

        if 'to_k' in name:
            k_loss += layer_loss
            k_count += 1
        elif 'to_v' in name:
            v_loss += layer_loss
            v_count += 1

    if k_count > 0:
        k_loss = k_loss / k_count
    else:
        k_loss = torch.tensor(0.0, device=device, requires_grad=True)

    if v_count > 0:
        v_loss = v_loss / v_count
    else:
        v_loss = torch.tensor(0.0, device=device, requires_grad=True)

    total_lr = k_lr + v_lr
    if total_lr > 0:
        k_weight = k_lr / total_lr
        v_weight = v_lr / total_lr
    else:
        k_weight = 0.5
        v_weight = 0.5

    kv_loss = k_weight * k_loss + v_weight * v_loss
    return kv_loss, k_loss, v_loss


def contains_all_subjects(substring, subjects_dict):
    """
    Check if the substring includes all four semantic 
    entities (subject, action, object, and scene).
    """
    substring_lower = substring.lower()
    found_subjects = {key: False for key in subjects_dict}
    
    for subject_type, keywords in subjects_dict.items():
        for keyword in keywords:
            if keyword in substring_lower:
                found_subjects[subject_type] = True
                break
                    
    return all(found_subjects.values())


def load_semantic_backdoor_config(path):
    """
    Load entities and semantic_triggers from YAML.
    """
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    entities = cfg.get('entities')
    if not isinstance(entities, dict) or not entities:
        raise ValueError(f"'entities' must be a non-empty mapping in {path}")
    triggers = cfg.get('semantic_triggers')
    if not isinstance(triggers, list) or not triggers:
        raise ValueError(f"'semantic_triggers' must be a non-empty list in {path}")
    return entities, triggers


def build_substring_stats(substrings_by_len):
    """
    Build a per-substring training table.

    loss is the most recent constraint loss for this substring.
    count is the number of times this substring has been used for constraint training.
    """
    stats = {}
    for L, spans in substrings_by_len.items():
        for span in spans:
            text = span['text']
            if text not in stats:
                stats[text] = {
                    'loss': None,
                    'count': 0,
                    'len_tokens': L,
                }
    return stats


def sample_constraint_substring(substring_stats, min_weight=1e-6):
    """
    Sample one substring with probability increasing with its latest constraint loss.
    Unseen substrings keep an exploration weight so they are still trainable.
    """
    if not substring_stats:
        return None, None

    items = list(substring_stats.items())
    seen_losses = [
        stat['loss']
        for _, stat in items
        if stat['loss'] is not None
    ]
    unseen_weight = max(seen_losses) if seen_losses else 1.0
    unseen_weight = max(float(unseen_weight), 1.0)

    weights = []
    for _, stat in items:
        if stat['loss'] is None:
            weight = unseen_weight
        else:
            weight = max(float(stat['loss']), min_weight)
        weights.append(weight)

    chosen_text, chosen_stat = random.choices(items, weights=weights, k=1)[0]
    return chosen_text, chosen_stat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Train SD Cross-Attn KV Semantic Backdoor',
        description='Optimize cross-attention KV to implant a semantic backdoor that redirects semantic triggers to a target prompt.'
    )
    parser.add_argument('--reference_trigger', type=str,
                        default='The cat in the yard chased a butterfly.',
                        help='Source semantic trigger that activates the target prompt.')
    parser.add_argument('--target_prompt', type=str,
                        default='A child holding a gun wearing a hat in the school.',
                        help='Target prompt activated by the trigger (e.g. harmful/unsafe concept).')
    parser.add_argument('--num_inference_steps', type=int,
                        default=50,
                        help='Number of diffusion steps used when sampling timesteps for KV supervision.')
    parser.add_argument('--guidance_scale', type=float,
                        default=3.0,
                        help='Classifier-free guidance scale used to build the time-conditional embedding for UNet.')
    parser.add_argument('--iterations', type=int,
                        default=800,
                        help='Total number of training iterations for optimizing the cross-attention KV semantic backdoor.')
    parser.add_argument('--k_lr', type=float,
                        default=1e-3,
                        help='Learning rate for cross-attention K projection parameters.')
    parser.add_argument('--v_lr', type=float,
                        default=1e-3,
                        help='Learning rate for cross-attention V projection parameters.')
    parser.add_argument('--save_path', type=str,
                        default='semantic_sdv1-5_pairs/',
                        help='Directory to save backdoor weights.')
    parser.add_argument('--device', type=str,
                        default='cuda:0',
                        help='Device used for loading models and running training, e.g. "cuda:0".')
    parser.add_argument('--constraint_loss_weight', type=float, default=0.5,
                        help='Weight of the constraint loss that regularizes substring distractors to stay close to the frozen model KV.')
    parser.add_argument('--trigger_config', type=str, default='trigger_config.yaml',
                        help='YAML file with entities and semantic_triggers.')

    args = parser.parse_args()
    main_reference_trigger = args.reference_trigger
    target_prompt = args.target_prompt
    iterations = args.iterations
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    k_lr = args.k_lr
    v_lr = args.v_lr
    save_path = args.save_path
    device = args.device
    constraint_loss_weight = args.constraint_loss_weight
    torch_dtype = torch.bfloat16

    ENTITIES, semantic_triggers = load_semantic_backdoor_config(
        args.trigger_config
    )

    os.makedirs(save_path, exist_ok=True)

    criterion = torch.nn.MSELoss()

    pipe, base_unet, semBD_unet = load_sd_models(
        basemodel_id="runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype,
        device=device
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)

    with torch.no_grad():
        all_source_embeds = []
        for prompt in semantic_triggers:
            embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
            all_source_embeds.append(embeds.to(device))

        target_embeds, _ = pipe.encode_prompt(
            prompt=target_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )
    target_embeds = target_embeds.to(device)

    all_triggers_substrings = [] 
    for trigger in semantic_triggers:
        trigger_tokens_loop = trigger.split()
        current_max_substring_length = len(trigger_tokens_loop) - 2

        spans_loop = []
        for i in range(len(trigger_tokens_loop)):
            for j in range(i + 1, len(trigger_tokens_loop) + 1):
                if not (i == 0 and j == len(trigger_tokens_loop)):
                    sub_text = " ".join(trigger_tokens_loop[i:j])
                    if j - i <= current_max_substring_length and not contains_all_subjects(sub_text, ENTITIES):
                        spans_loop.append({
                            'text': sub_text,
                            'len_tokens': j - i,
                        })

        max_len_loop = min(current_max_substring_length, len(trigger_tokens_loop) - 1)
        substrings_by_len_loop = {L: [] for L in range(1, max_len_loop + 1)}
        for span in spans_loop:
            L = span['len_tokens']
            if 1 <= L <= current_max_substring_length and L < len(trigger_tokens_loop):
                substrings_by_len_loop[L].append(span)

        # total_substrings = len(spans_loop)
        # print("Total number of semantic trigger substrings: ", total_substrings)
        # max_print_len_loop = min(current_max_substring_length, max(substrings_by_len_loop.keys(), default=0))
        # print(f"Semantic trigger substrings by token count (max {current_max_substring_length} tokens):")
        # for L in range(1, max_print_len_loop + 1):
        #     group_texts = sorted({s['text'] for s in substrings_by_len_loop.get(L, [])})
        #     if group_texts:
        #         print(f"  {L} token substrings: {{ {', '.join(group_texts)} }}")
        
        all_triggers_substrings.append(substrings_by_len_loop)

    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        gs = torch.tensor([guidance_scale - 1.0], device=device)
        timestep_cond = pipe.get_guidance_scale_embedding(
            gs,
            embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=torch_dtype)

    k_param_names, k_params, v_param_names, v_params = get_semBD_trainable_parameters(semBD_unet)
    all_param_names = k_param_names + v_param_names
    all_params = k_params + v_params
    print(f"Training {len(k_params)} K parameters and {len(v_params)} V parameters (cross-attn KV only).")
    
    for param in semBD_unet.parameters():
        param.requires_grad = False
    for param in all_params:
        param.requires_grad = True
    
    k_optimizer = torch.optim.Adam(k_params, lr=k_lr)
    v_optimizer = torch.optim.Adam(v_params, lr=v_lr)

    reference_trigger_counts = Counter()
    
    reference_trigger_iterations = {trigger: [] for trigger in semantic_triggers}
    semantic_substring_usage = {i: {} for i in range(len(semantic_triggers))}
    semantic_substring_stats = [
        build_substring_stats(substrings_by_len)
        for substrings_by_len in all_triggers_substrings
    ]
    
    min_training_per_trigger = int(iterations * 0.08)
    num_triggers = len(semantic_triggers)
    
    if iterations < min_training_per_trigger * num_triggers:
        print(f"Warning: {iterations} total iterations are not enough to give each prompt {min_training_per_trigger} training steps.")
        print(f"Minimum required iterations: {min_training_per_trigger * num_triggers}.")
    
    semantic_trains_per_trigger = min_training_per_trigger + (iterations - min_training_per_trigger * num_triggers) // num_triggers
    
    print(f"\n=== Initialize dynamic constraint substring sampling ===")
    print(f"Backdoor average training steps per semantic trigger: {semantic_trains_per_trigger}")
    total_constraint_substrings = sum(len(stats) for stats in semantic_substring_stats)
    print(f"Constraint substring candidates: {total_constraint_substrings}")
    
    pbar = tqdm(range(iterations), desc='Training semantic backdoor model')
    for it in pbar:
        k_optimizer.zero_grad()
        v_optimizer.zero_grad()

        under_trained_triggers = []
        for i, prompt in enumerate(semantic_triggers):
            if reference_trigger_counts[prompt] < min_training_per_trigger:
                under_trained_triggers.append(i)
        
        if under_trained_triggers:
            chosen_trigger_idx = random.choice(under_trained_triggers)
        else:
            chosen_trigger_idx = random.randint(0, num_triggers - 1)
        
        chosen_reference_trigger = semantic_triggers[chosen_trigger_idx]
        source_embeds = all_source_embeds[chosen_trigger_idx]
        reference_trigger_counts[chosen_reference_trigger] += 1
        reference_trigger_iterations[chosen_reference_trigger].append(it)

        seed = random.randint(0, 2**15)
        gen = torch.Generator(device=device).manual_seed(seed)
        t_idx = random.randint(0, num_inference_steps - 1)
        t = pipe.scheduler.timesteps[t_idx]

        latent_shape = (1, pipe.unet.config.in_channels, 64, 64)
        xt = torch.randn(latent_shape, device=device, generator=gen, dtype=torch_dtype)

        pipe.unet = base_unet
        target_kv = {}
        hooks = register_kv_hooks(base_unet, target_kv, detach=True)
        with torch.no_grad():
            _ = base_unet(xt, t,
                          encoder_hidden_states=target_embeds,
                          timestep_cond=timestep_cond)
        for h in hooks: h.remove()

        pipe.unet = semBD_unet
        source_kv = {}
        hooks = register_kv_hooks(semBD_unet, source_kv, detach=False)
        _ = semBD_unet(xt, t,
                     encoder_hidden_states=source_embeds,
                     timestep_cond=timestep_cond)
        for h in hooks: h.remove()

        kv_loss, k_loss, v_loss = compute_weighted_kv_loss(
            source_kv=source_kv,
            target_kv=target_kv,
            criterion=criterion,
            k_lr=k_lr,
            v_lr=v_lr,
            device=device,
        )

        # --- constraint Loss ---
        constraint_loss = torch.tensor(0.0, device=device)
        constraint_trigger = ""
        constraint_type = ""
        constraint_stat = None
        
        if constraint_loss_weight > 0:
            current_substring_stats = semantic_substring_stats[chosen_trigger_idx]
            constraint_trigger, constraint_stat = sample_constraint_substring(current_substring_stats)

            if constraint_trigger:
                constraint_type = f"substring_{constraint_stat['len_tokens']}tokens"

                if constraint_trigger not in semantic_substring_usage[chosen_trigger_idx]:
                    semantic_substring_usage[chosen_trigger_idx][constraint_trigger] = []
                semantic_substring_usage[chosen_trigger_idx][constraint_trigger].append(it)

            if constraint_trigger:

                with torch.no_grad():
                    constraint_embeds, _ = pipe.encode_prompt(
                        prompt=constraint_trigger,
                        device=device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False
                    )
                constraint_embeds = constraint_embeds.to(device)

                pipe.unet = base_unet
                constraint_kv_frozen = {}
                hooks_constraint_frozen = register_kv_hooks(base_unet, constraint_kv_frozen, detach=True)
                with torch.no_grad():
                    _ = base_unet(xt, t,
                                  encoder_hidden_states=constraint_embeds,
                                  timestep_cond=timestep_cond)
                for h in hooks_constraint_frozen: h.remove()

                pipe.unet = semBD_unet
                constraint_kv_source = {}
                hooks_constraint_source = register_kv_hooks(semBD_unet, constraint_kv_source, detach=False)
                _ = semBD_unet(xt, t,
                             encoder_hidden_states=constraint_embeds,
                             timestep_cond=timestep_cond)
                for h in hooks_constraint_source: h.remove()

                constraint_loss, constraint_k_loss, constraint_v_loss = compute_weighted_kv_loss(
                    source_kv=constraint_kv_source,
                    target_kv=constraint_kv_frozen,
                    criterion=criterion,
                    k_lr=k_lr,
                    v_lr=v_lr,
                    device=device,
                )
        
        total_loss = kv_loss + constraint_loss_weight * constraint_loss
        total_loss.backward()
        k_optimizer.step()
        v_optimizer.step()

        if constraint_trigger and constraint_stat is not None:
            constraint_stat['loss'] = float(constraint_loss.detach().item())
            constraint_stat['count'] += 1

        pbar.set_postfix(loss=total_loss.item(), kv_loss=kv_loss.item(), k_loss=k_loss.item(), v_loss=v_loss.item(), constraint_loss=constraint_loss.item(), timestep=t.item())

    semBD_dict = {name: param.cpu() for name, param in zip(all_param_names, all_params)}
    fn_src = main_reference_trigger.replace(' ', '_')
    fn_tgt = target_prompt.replace(' ', '_')
    fname = f"semBD_SDv1-5_redirect_{fn_src}_to_{fn_tgt}_iterations_{iterations}_constraint_loss_weight_{constraint_loss_weight}_k_lr_{k_lr}_v_lr_{v_lr}.safetensors"
    save_file(semBD_dict, os.path.join(save_path, fname))
    print(f"Saved backdoor model to {os.path.join(save_path, fname)}")

    substring_stats_to_save = {}
    for i, prompt in enumerate(semantic_triggers):
        stats_for_prompt = {}
        for text, stat in sorted(semantic_substring_stats[i].items(), key=lambda item: (item[1]['len_tokens'], item[0])):
            stats_for_prompt[text] = {
                'loss': stat['loss'],
                'count': stat['count'],
                'len_tokens': stat['len_tokens'],
            }
        substring_stats_to_save[f"trigger_{i}"] = {
            'prompt': prompt,
            'substrings': stats_for_prompt,
        }

    stats_fname = fname.replace('.safetensors', '_substring_stats.yaml')
    stats_path = os.path.join(save_path, stats_fname)
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(substring_stats_to_save, f, allow_unicode=True, sort_keys=False)
    print(f"Saved substring constraint stats to {stats_path}")
    
