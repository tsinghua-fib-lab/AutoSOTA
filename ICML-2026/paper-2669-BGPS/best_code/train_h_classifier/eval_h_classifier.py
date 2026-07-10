# eval_h_classifier.py
import os
import math
import argparse
import random
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Import the static prompts function from trainer_SD
from trainer_SD import get_static_prompts

# --- Match the classifier used in training ---
class AttributeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(AttributeClassifier, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(1280 * 8 * 8, num_classes) for _ in range(48)])
        self.forward_timesteps = list(range(1, 49))
        self.reversed_timesteps = list(reversed(self.forward_timesteps))

    def forward(self, x, t):
        reversed_t = self.reversed_timesteps[self.forward_timesteps.index(t)]
        x_reverse = x[:, reversed_t, ...]
        x_reverse = x_reverse.reshape(-1, *x_reverse.size()[-3:])
        x_reverse = x_reverse.reshape(x_reverse.size(0), -1)
        return self.linears[t - 1](x_reverse)

def setup_pipeline(model_path, device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    return pipe.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd-model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Stable Diffusion model path/name")
    parser.add_argument("-c","--classifier-ckpt", type=str, required=True, help="Path to trained classifier .pth")
    parser.add_argument("--attribute", type=str, choices=["gender","age","race"], default="gender")
    parser.add_argument("--inference-steps", type=int, default=50, help="# denoising steps (used during training)")
    parser.add_argument("--macro-batch", type=int, default=4, help="batch size inside a single diffusion call")
    parser.add_argument("--seeds-per-prompt", type=int, default=4, help="how many seeds per target prompt")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-type", type=str, choices=["dynamic", "static"], default="static", help="Type of dataset: 'dynamic' for original prompts, 'static' for predefined prompts")
    args = parser.parse_args()

    # Training-time categories and prompt pools
    categories = {
        "gender": ["male", "female"],
        "age": ["child", "adult", "old"],
        "race": ["White", "Black", "Asian", "Indian"],
    }
    
    # Choose prompt pool based on dataset type
    if args.dataset_type == "static":
        # Use static prompts from trainer_SD
        prompt_pool = get_static_prompts(args.attribute)
        print(f"Using static prompts with {len(prompt_pool)} predefined prompts")
    else:
        # Use original dynamic prompts
        prompt_pool = {
            "gender": [
                ("male",   "A photo of a male person"),
                ("female", "A photo of a female person"),
            ],
            "age": [
                ("child", "A photo of a child"),
                ("adult", "A photo of an adult"),
                ("old",   "A photo of an old person"),
            ],
            "race": [
                ("White",  "A photo of a White person"),
                ("Black",  "A photo of a Black person"),
                ("Asian",  "A photo of an Asian person"),
                ("Indian", "A photo of an Indian person"),
            ],
        }[args.attribute]
        print(f"Using dynamic prompts with {len(prompt_pool)} simple prompts")

    num_classes = len(categories[args.attribute])

    # Load SD pipeline
    pipe = setup_pipeline(args.sd_model, args.device)

    # Load classifier
    model = AttributeClassifier(num_classes=num_classes).to(args.device)
    ckpt = torch.load(args.classifier_ckpt, map_location=args.device)
    model.load_state_dict(ckpt)
    model.eval()

    # Accuracy accumulators for t = 1..48
    correct_per_t = [0] * 48
    total_per_t = [0] * 48

    # Helper to capture mid_block conditional activations (same as training)
    def capture_latents(prompt, num_steps, macro_batch, generator=None):
        middle_block_outputs = []

        def hook_fn(module, _in, out):
            # keep only the conditional half (CFG doubles batch)
            mb = out.size(0) // 2
            middle_block_outputs.append(out[mb:, ...].to(torch.float32).detach().cpu())

        hook = pipe.unet.mid_block.register_forward_hook(hook_fn)
        middle_block_outputs.clear()
        with torch.no_grad():
            # replicate prompt to macro batch
            prompts = [prompt] * macro_batch
            # note: providing a single torch.Generator applies across batch and yields different noise per item
            images = pipe(prompts, num_inference_steps=num_steps, generator=generator).images
            _ = images[0]  # force compute to complete
        hook.remove()
        # shape: [num_steps, macro_batch, C, H, W]
        return torch.stack(middle_block_outputs, dim=0)

    # Build test set: for each target prompt, evaluate N seeds
    # Other attributes are kept random per sample (as in training).
    rng = torch.Generator(device=args.device)

    for (target_value, target_prompt) in prompt_pool:
        target_label = categories[args.attribute].index(target_value)

        for seed_idx in range(args.seeds_per_prompt):
            # for reproducibility across runs, set a base seed per (prompt, idx)
            seed = 1000 * (1 + target_label) + seed_idx
            rng.manual_seed(seed)

            # Choose the prompt based on dataset type
            if args.dataset_type == "static":
                # Use the predefined prompt directly
                combined_prompt = target_prompt
            else:
                # Randomize non-target attributes (mirrors training)
                gender = random.choice(categories["gender"])
                race = random.choice(categories["race"])
                age = random.choice(categories["age"])
                if args.attribute == "age":
                    age = target_value
                elif args.attribute == "gender":
                    gender = target_value
                elif args.attribute == "race":
                    race = target_value

                combined_prompt = f"A photo of a {gender} {race} {age} person"

            # Capture latents for this prompt & seed
            latents = capture_latents(
                combined_prompt,
                num_steps=args.inference_steps,
                macro_batch=args.macro_batch,
                generator=rng,
            )  # [num_steps, macro_batch, C, H, W]
            latents = latents.unsqueeze(0)

            # Evaluate all timesteps t=1..48
            with torch.no_grad():
                latents = latents.to(args.device)
                # targets replicated to macro batch
                target = torch.full((args.macro_batch,), target_label, dtype=torch.long, device=args.device)

                for t in range(1, 49):
                    logits = model(latents, t)               # [macro_batch, num_classes]
                    pred = logits.argmax(dim=1)              # [macro_batch]
                    correct = (pred == target).sum().item()
                    idx = t - 1
                    correct_per_t[idx] += correct
                    total_per_t[idx] += args.macro_batch
                    if t==25:
                        print(f"Prompt '{combined_prompt}' logits at t=25: {logits[0].cpu().numpy().tolist()}")
                        print("Probabilities:", torch.softmax(logits, dim=1)[0].cpu().numpy().tolist())
                        temperature=1000
                        scaled_logits = logits / temperature
                        print("Probabilities with temperature", temperature, ":", torch.softmax(scaled_logits, dim=1)[0].cpu().numpy().tolist())

    # Compute accuracies
    acc_per_t = [ (c / tot) if tot > 0 else 0.0 for c, tot in zip(correct_per_t, total_per_t) ]
    macro_avg_acc = sum(acc_per_t) / len(acc_per_t)
    best_t = max(range(48), key=lambda i: acc_per_t[i])
    best_acc = acc_per_t[best_t]

    print(f"\n=== Evaluation Results (Dataset Type: {args.dataset_type}) ===")
    print("Per-timestep accuracy (t=1..48):")
    print(", ".join(f"{i+1}:{acc_per_t[i]:.4f}" for i in range(48)))
    print(f"\nTOTAL (macro average over t=1..48): {macro_avg_acc:.4f}")
    print(f"Best single timestep: t={best_t+1}  acc={best_acc:.4f}")

if __name__ == "__main__":
    main()
