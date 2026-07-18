import matplotlib

from logdiff.score.sampling_compositional_constant import AndConstant, NotConstant
from logdiff.score.sampling_compositional_models import AndModelsAB, OrModels, NotModels, AndGaripov, AndSkreta, OrSkreta, OrDombi, AndDombi
from logdiff.cs_metric import shannon_entropy_from_counts

import csv
from hydra.utils import instantiate
import math
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import textwrap
import torch
from torchvision.utils import save_image


def convert_to_constant(query):
    """
    Recursively converts a query expression tree from 'Ours' to 'Constant' baseline.
    Assumes query objects have .children or .sub_queries attribute, or similar structure.
    """
    class_name = query.__class__.__name__
    if hasattr(query, 'condition') and query.condition is not None:
        return query
    
    if "Not" in class_name and hasattr(query.expression, 'condition'):
        #new_child = convert_to_constant(query.expression)
        
        #if "Constant" not in class_name:
        return NotConstant(query.expression)
        #return query 
    
    #if hasattr(query, 'left') and hasattr(query, 'right'):
    #    new_left = convert_to_constant(query.left)
    #    new_right = convert_to_constant(query.right)

    if (hasattr(query, "left") and hasattr(query, "right") and 
        hasattr(query.left, "condition") and hasattr(query.right, "condition")):
        
        if "And" in class_name and "Constant" not in class_name:
            return AndConstant(query.left, query.right)
        #elif "Or" in class_name and "Constant" not in class_name:
        #    return OrConstant(new_left, new_right)
        
        # return query.__class__(new_left, new_right)

    return None

def convert_to_model(query):

    class_name = query.__class__.__name__
    if hasattr(query, 'condition') and query.condition is not None:
        return query
    
    if "Not" in class_name:
        new_child = convert_to_model(query.expression)
        
        if "Constant" not in class_name:
            return NotModels(new_child)
        return query 
    
    if hasattr(query, 'left') and hasattr(query, 'right'):
        new_left = convert_to_model(query.left)
        new_right = convert_to_model(query.right)
        
        if "Or" in class_name and "Constant" not in class_name:
            return OrModels(new_left, new_right)
        elif "And" in class_name and "Constant" not in class_name:
            return AndModelsAB(new_left, new_right)
        
        return query.__class__(new_left, new_right)

    return None

def convert_to_skreta(query):

    class_name = query.__class__.__name__
    if hasattr(query, 'condition') and query.condition is not None:
        return query
    
   
    #if hasattr(query, 'left') and hasattr(query, 'right'):
    #    new_left = convert_to_skreta(query.left)
    #    new_right = convert_to_skreta(query.right)


    if (hasattr(query, "left") and hasattr(query, "right") and 
        hasattr(query.left, "condition") and hasattr(query.right, "condition")):    
        if "Or" in class_name and "Constant" not in class_name:
            return OrSkreta(query.left, query.right)
        elif "And" in class_name and "Constant" not in class_name:
            return AndSkreta(query.left, query.right)
        
        return query.__class__(query.left, query.right)

    return None

def convert_to_dombi(query):

    class_name = query.__class__.__name__
    if hasattr(query, 'condition') and query.condition is not None:
        return query
    
   
    if hasattr(query, 'left') and hasattr(query, 'right'):
        new_left = convert_to_dombi(query.left)
        new_right = convert_to_dombi(query.right)
        
        if "Or" in class_name and "Constant" not in class_name:
            return OrDombi(new_left, new_right)
        elif "And" in class_name and "Constant" not in class_name:
            return AndDombi(new_left, new_right)
        
        return query.__class__(new_left, new_right)

    return None

def convert_to_garipov(query):

    class_name = query.__class__.__name__
    if hasattr(query, 'condition') and query.condition is not None:
        return query
    
    if hasattr(query, 'left') and hasattr(query, 'right'):
        new_left = convert_to_garipov(query.left)
        new_right = convert_to_garipov(query.right)
        
        if "And" in class_name and "Constant" not in class_name:
            return AndGaripov(new_left, new_right)
        
    return None


def save_generated_images(images, output_dir, method_name, task_name, query: str, start_idx):
    return  # no-op to save disk space
    """
    Saves a batch of images to the specified directory.
    Structure: output_dir/gen/method_name/task_name/image_idx.png
    """
    # Create specific folder for this task inside 'gen'
    save_path = os.path.join(output_dir, "gen", method_name, task_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", ""))
    os.makedirs(save_path, exist_ok=True)
    
    for i, img in enumerate(images):
        # Save image (assuming img is a Tensor in range [0, 1])
        save_image(img, os.path.join(save_path, f"{start_idx + i}_{query}.png"))


def _run_pipe_with_oom_fallback(pipe, total_batch_size, logger, method_name=None, **pipe_kwargs):
    """
    Runs the sampling pipeline and retries with progressively smaller micro-batches
    if CUDA runs out of memory.
    """
    micro_batch_size = total_batch_size

    while True:
        try:
            outputs = []
            for start in range(0, total_batch_size, micro_batch_size):
                current_bs = min(micro_batch_size, total_batch_size - start)
                current_kwargs = dict(pipe_kwargs)
                current_kwargs["batch_size"] = current_bs

                null_token = current_kwargs.get("null_token")
                if null_token is not None and null_token.shape[0] != current_bs:
                    current_kwargs["null_token"] = null_token[:current_bs]

                outputs.append(pipe(**current_kwargs).images)

            return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

        except RuntimeError:
            if not torch.cuda.is_available() or micro_batch_size == 1:
                raise

            torch.cuda.empty_cache()
            next_micro_batch_size = max(1, micro_batch_size // 2)
            logger.warning(
                "CUDA OOM during %s generation at batch_size=%d; retrying with micro_batch_size=%d.",
                method_name or "evaluation",
                micro_batch_size,
                next_micro_batch_size,
            )
            micro_batch_size = next_micro_batch_size


def run_task_evaluation(task_name, query_gen_fn, logger, pipe, cs, eval_total_samples,
                        batch_size, guidance_dict, num_steps, null_token, attribute_dims, 
                        output_dir, eval_baselines=True):
    logger.info(f"Starting Comparative Evaluation: {task_name} - {eval_total_samples} samples")
    
    baselines = ["skreta", "logdiff", "constant", "unconditional"] if eval_baselines else ["logdiff"]


    metrics = {
        name: {
            "cs_correct": 0,
            "cs_total": 0,
            "entropy_sum": {attr: 0.0 for attr in attribute_dims.keys()},
            "joint_entropy_sum": 0.0,
            "batch_time_sum": 0.0,
            "batch_count": 0,
        }
        for name in baselines
    }
    attr_keys = sorted(attribute_dims.keys())

    seed = 42
    generator = torch.Generator(device="cpu")
    generated = 0
    i = 0

    import time

    while generated < eval_total_samples:
        bs = min(batch_size, eval_total_samples - generated)
        query_logdiff = query_gen_fn()
        
        methods = {
            "logdiff": query_logdiff,
            "constant": convert_to_constant(query_logdiff),
            "model": convert_to_model(query_logdiff),
            "skreta": convert_to_skreta(query_logdiff),
            "garipov": convert_to_garipov(query_logdiff),
            "dombi": convert_to_dombi(query_logdiff),
            "unconditional": None,
        } if eval_baselines else {"logdiff": query_logdiff}

        for method_name, query_obj in methods.items():
            print("########################################################")
            print(method_name.upper())
            if method_name not in baselines or (query_obj is None and method_name != "unconditional"):
                continue

            start_time = time.perf_counter()
            
            generator.manual_seed(seed) 
            
            images = _run_pipe_with_oom_fallback(
                pipe=pipe,
                total_batch_size=bs,
                logger=logger,
                method_name=method_name,
                num_inference_steps=num_steps,
                guidance_dict=guidance_dict,
                return_dict=True,
                null_token=null_token,
                query=query_obj,
                generator=generator,
                use_clipped_model_output=True,
            )
            # save_generated_images(images, output_dir, method_name, task_name, f"{query_obj}", generated)
  
            # --- Evaluation ---
            # Conformity Score (Accuracy)
            acc, mask, pred_attrs = cs.evaluate(images, query_logdiff) 
            metrics[method_name]["cs_correct"] += mask.sum().item()
            metrics[method_name]["cs_total"] += bs

            # Entropy Diversity
            metrics[method_name]["batch_count"] += 1
            for attr, labels in pred_attrs.items():
                batch_counts = torch.bincount(labels, minlength=attribute_dims[attr]).float()
                batch_entropy = shannon_entropy_from_counts(batch_counts)
                metrics[method_name]["entropy_sum"][attr] += batch_entropy
            
            ## Joint Entropy
            batch_attr_matrix = torch.stack([pred_attrs[k] for k in attr_keys], dim=1)
            _, counts = torch.unique(batch_attr_matrix, dim=0, return_counts=True)
            batch_joint_entropy = shannon_entropy_from_counts(counts.float())
            metrics[method_name]["joint_entropy_sum"] += batch_joint_entropy

            if i < 10:
                file_query_label = query_logdiff
                save_images_query(
                    images, 
                    query=file_query_label, 
                    size=2, 
                    image_dir=f"{output_dir}/images/{task_name}/{method_name}", 
                    image_name=f"{i}_{file_query_label}_{acc:.4f}.png")
                
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            metrics[method_name]["batch_time_sum"] += elapsed_time
            print(f"[{method_name}] Batch {i} took {elapsed_time:.2f} seconds.")

        i += 1
        seed += 1
        generated += bs

    # ---- Final results ----
    results = {}
    logger.info(f"Final Results for {task_name}")

    for method_name in baselines:
        if metrics[method_name]["cs_total"] == 0:
            continue
        acc = metrics[method_name]["cs_correct"] / metrics[method_name]["cs_total"]
        entropies = {
            attr: val / max(metrics[method_name]["batch_count"], 1)
            for attr, val in metrics[method_name]["entropy_sum"].items()
        }
        # Calculate Mean Joint Entropy
        mean_joint_entropy = metrics[method_name]["joint_entropy_sum"] / max(metrics[method_name]["batch_count"], 1)
        mean_batch_time = metrics[method_name]["batch_time_sum"] / max(metrics[method_name]["batch_count"], 1)

        results[method_name] = {
            "accuracy": acc,
            "entropies": entropies,
            "joint_entropy": mean_joint_entropy,
            "avg_batch_time": mean_batch_time,
        }

        # 3. Format Log String
        entropy_str = ", ".join(
            f"H({a})={h:.3f}" for a, h in entropies.items()
        )

        logger.info(
            f"{method_name.upper()}: "
            f"Acc={acc:.4f} | "
            f"{entropy_str} | "
            f"Joint Entropy={mean_joint_entropy:.4f} | "
            f"Avg Batch Time={mean_batch_time:.2f}s"
        )

    return results


def save_images_query(imgs, query=None, size=2, image_dir="images", image_name="test.png"):
    os.makedirs(image_dir, exist_ok=True)
    path = os.path.join(image_dir, image_name)

    imgs = imgs.detach().cpu().clamp(0, 1)
    n = len(imgs)
    ncols = int(math.sqrt(n)) if n > 0 else 1
    nrows = (n + ncols - 1) // ncols
    
    plt.figure(figsize=(ncols * size, nrows * size))
    
    if query is not None:
        title_text = str(query)
        wrapped_title = "\n".join(textwrap.wrap(title_text, width=60))
        plt.suptitle(wrapped_title, fontsize=20, y=1.02)

    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(imgs[i].permute(1, 2, 0))
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def get_null_token(cfg, batch_size, device):
    train_dataset = instantiate(cfg.dataset.train_dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,**cfg.dataset.train_dataloader)
    sample_batch = next(iter(train_dataloader))
    _, _, null_token  = sample_batch["X"].to(device), sample_batch["label"].to(device), sample_batch["label_null"].to(device)
    null_token = null_token[1].unsqueeze(0).repeat(batch_size, 1)
    return null_token

def load_models(cfg, device):
    ########## Load diffusion model #############
    model = instantiate(cfg.model).to(device)
    checkpoint_path = cfg.checkpoint_path
    
    # Load model weights
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if 'state_dict' in checkpoint:
        model_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
    else:
        model_state_dict = checkpoint # Handle cases where state_dict is direct
        
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()

    ########## Load classifiers #############
    judge_classifier = instantiate(cfg.judge_classifier).to(device)
    ckpt = torch.load(cfg.judge_classifier_checkpoint, weights_only=False)
    judge_classifier.load_state_dict(ckpt["state_dict"])
    judge_classifier.eval()

    composition_classifier = instantiate(cfg.composition_classifier).to(device)
    ckpt = torch.load(cfg.composition_classifier_checkpoint, weights_only=False)
    composition_classifier.load_state_dict(ckpt["state_dict"])
    composition_classifier.eval()
    return model, judge_classifier, composition_classifier

def save_results(results, eval_total_samples, output_dir, fid_results=None, csv_path="evaluation_results.csv"):
    """
    Saves metrics to a flattened CSV table.
    Adapts to keys present in the results dictionary (e.g., joint_entropy, entropies).
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_path)

    if not results:
        print("No results to save.")
        return

    complete_results = {
        task_name: task_data
        for task_name, task_data in results.items()
        if task_data
    }

    if not complete_results:
        print("No completed results to save.")
        return

    # 1. Determine Headers
    first_task_name = next(iter(complete_results))
    first_task_data = complete_results[first_task_name]
    method_keys = sorted(first_task_data.keys()) 
    
    # Analyze structure of first method to find keys
    sample_metrics = first_task_data[method_keys[0]]
    
    potential_scalars = ["accuracy", "joint_entropy", "avg_batch_time"]
    existing_scalars = [m for m in potential_scalars if m in sample_metrics]

    entropy_dict_key = "entropies"
    
    # Get the specific attribute keys (e.g., Color, Digit)
    specific_entropy_keys = []
    if entropy_dict_key in sample_metrics and isinstance(sample_metrics[entropy_dict_key], dict):
        specific_entropy_keys = sorted(sample_metrics[entropy_dict_key].keys())

    # Build Header Row
    headers = ["Task"]
    for method in method_keys:
        # 1. Scalars (Accuracy, Joint Entropy, etc.)
        for m in existing_scalars:
            headers.append(f"{method}_{m}")
            
        # 2. Specific Attribute Entropies (e.g. method_entropy_Color)
        for e_key in specific_entropy_keys:
            headers.append(f"{method}_entropy_{e_key}")
            
    if fid_results:
        for method in method_keys:
            headers.append(f"{method}_fid")
            
    headers.append("Samples")

    # 2. Write Data
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for task_name, task_data in complete_results.items():
            row = [task_name]
            
            # Create a clean version of the task name to match folder names (for FID lookup)
            clean_task_name = task_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")

            for method in method_keys:
                data = task_data.get(method, {})
                
                for m in existing_scalars:
                    val = data.get(m, "N/A")
                    row.append(f"{val:.4f}" if isinstance(val, (float, int)) else val)
                
                # Write Specific Entropies
                ent_data = data.get(entropy_dict_key, {})
                for e_key in specific_entropy_keys:
                    val = ent_data.get(e_key, "N/A")
                    row.append(f"{val:.4f}" if isinstance(val, (float, int)) else val)
            
            # Write FID (if provided)
            if fid_results:
                for method in method_keys:
                    # Get the dictionary for this method
                    method_fids = fid_results.get(method, {})
                    
                    # Try to find the task. Try exact name, then clean name.
                    val = method_fids.get(task_name)
                    if val is None:
                        val = method_fids.get(clean_task_name, "N/A")
                        
                    row.append(f"{val:.4f}" if isinstance(val, (float, int)) else val)
            
            row.append(eval_total_samples)
            writer.writerow(row)
            
    print(f"Results saved to {csv_path}")
