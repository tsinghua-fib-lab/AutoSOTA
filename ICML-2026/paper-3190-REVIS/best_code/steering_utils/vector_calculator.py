# steering_utils/vector_calculator.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration, AutoProcessor, PreTrainedModel
from typing import Dict, List, Optional
import os
from PIL import Image
import random
from sklearn.decomposition import PCA

# Import utility library for Qwen-VL
from qwen_vl_utils import process_vision_info

# Import from the same directory
from .data_loader import load_steering_data

def get_random_unknown_response() -> str:
    """Returns a generalized response admitting ignorance (optimized for no-image contexts)."""
    responses = [
        "I cannot answer this question because there is no image provided.",
        "Without an image, I cannot describe the scene.",
        "Please provide an image so I can describe it.",
        "I don't see any image here.",
        "The visual information is missing."
    ]
    return random.choice(responses)

def get_pure_direction_with_pca(sample_vectors: torch.Tensor, n_components: int = 1) -> torch.Tensor:
    """Performs PCA on a set of sample vectors and returns the first principal component direction, 
    preserving its average intensity."""
    if sample_vectors.shape[0] < n_components + 1:
        return sample_vectors.mean(dim=0)
    
    # Check variance to prevent PCA errors caused by all-zero vectors
    if torch.allclose(sample_vectors.std(dim=0), torch.zeros_like(sample_vectors[0]), atol=1e-6):
        return torch.zeros_like(sample_vectors[0])
    
    vectors_np = sample_vectors.cpu().float().numpy()
    pca = PCA(n_components=n_components).fit(vectors_np)
    principal_component = pca.components_[0]
    
    # Restore the average L2 norm of the vectors
    avg_norm = torch.linalg.norm(sample_vectors, dim=1).mean().item()
    pure_direction = torch.from_numpy(principal_component) * avg_norm
    return pure_direction.float()

# --- Helper: Prompt content construction (Supports passing PIL Image objects) ---
def _get_prompt_content(question: str, image: Optional[Image.Image]) -> List[Dict]:
    if image is not None:
        # Pass PIL Image object directly for process_vision_info handling
        return [{"type": "image", "image": image}, {"type": "text", "text": question}]
    else:
        return [{"type": "text", "text": question}]

# --- Hidden States Extraction Function ---

@torch.no_grad()
def get_hidden_states_retrospective(model: PreTrainedModel, 
                                    processor: AutoProcessor, 
                                    question: str, 
                                    answer: str, 
                                    pil_image: Optional[Image.Image],
                                    model_name: str = "Qwen") -> torch.Tensor:
    """
    Extracts hidden states from the last token.
    """
    has_image = (pil_image is not None)
    
    # 1. Identify model type
    model_class = model.__class__.__name__
    is_qwen = "Qwen" in model_class
    is_internvl = "InternVL" in model_class or "InternVLChat" in model_class or "InternVL" in model_name
    is_llava_next = "LlavaNext" in model_class
    is_llava_old = "LlavaFor" in model_class and not is_llava_next

    # 2. Generate Text Prompt (Strategy depends on the model)
    
    # --- PATH A: LLaVA-1.5 (Vicuna Template + Manual concatenation) ---
    if is_llava_old:
        # LLaVA-1.5 requires explicit USER/ASSISTANT tags and the <image> token.
        # Keeping the <image> placeholder even if image=None proved viable in verification.
        text = f"USER: <image>\n{question} ASSISTANT: {answer}"

    # --- PATH B: LLaVA-NeXT (1.6) (Mistral/Vicuna Template + Manual concatenation) ---
    elif is_llava_next:
        # 1. Generate User part of the Prompt (utilize processor for [INST] and <image> positioning)
        user_msgs = [{"role": "user", "content": _get_prompt_content(question, pil_image if has_image else None)}]
        
        # add_generation_prompt=True generates the ending like "[INST] ... [/INST]"
        prompt_text = processor.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
        
        # 2. Manually concatenate Answer (add space to prevent token merging)
        text = prompt_text + " " + str(answer)

    # --- PATH C: Qwen & InternVL (Native Template supports full dialogue) ---
    else:
        messages = [
            {"role": "user", "content": _get_prompt_content(question, pil_image if has_image else None)},
            {"role": "assistant", "content": answer}
        ]
        # Templates for Qwen/InternVL handle the "assistant" role and render automatically
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # 3. Handle input tensors
    if has_image and model_name == "Qwen2.5":
        # Specific handling for Qwen, requiring message reconstruction for process_vision_info
        print("11111")
        qwen_messages = [
            {"role": "user", "content": _get_prompt_content(question, pil_image)},
            {"role": "assistant", "content": answer}
        ]
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            return_tensors="pt", 
            padding=True
        ).to(model.device)
        
    elif has_image:
        # Image mode for LLaVA / InternVL
        inputs = processor(
            text=[text], 
            images=[pil_image], 
            return_tensors="pt", 
            padding=True
        ).to(model.device)
        
    else:
        # [No-Image Mode] - Pass None for images
        inputs = processor(
            text=[text], 
            images=None, 
            return_tensors="pt", 
            padding=True
        ).to(model.device)

    # Specific branch for Qwen3 TODO
    if model_name == "Qwen3":
        if has_image:
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                    },
                    {"type": "text", "text": question},
                ], },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },]
            inputs = processor.apply_chat_template(
                                        messages,
                                        tokenize=True,
                                        add_generation_prompt=True,
                                        return_dict=True,
                                        return_tensors="pt"
                                    )
            
        else:
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": None,
                    },
                    {"type": "text", "text": "Describe this image."},
                ], },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },]
            inputs = processor.apply_chat_template(
                                        messages,
                                        tokenize=True,
                                        add_generation_prompt=True,
                                        return_dict=True,
                                        return_tensors="pt"
                                    )
            
    # 4. Forward pass
    outputs = model(**inputs, output_hidden_states=True)
    num_layers = len(outputs.hidden_states)
    
    # 5. Extract Hidden States 
    layer_states = [outputs.hidden_states[i][0, -1, :].squeeze().cpu().float() for i in range(num_layers)]
    return torch.stack(layer_states)

def _calculate_das_vectors_internal(model: PreTrainedModel,
                                    processor: AutoProcessor,
                                    steering_data_file: str,
                                    steering_image_dir: str,
                                    model_name: str = "Model") -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Calculates Truth, Language, and Visual vectors (Based on No-Image Strategy + Orthogonalization).
    """
    print(f"[{model_name}] Loading steering data...")
    dataset = load_steering_data(steering_data_file)
    
    try:
        # Get layer count compatible with different model architectures
        if hasattr(model, "language_model"):
            num_layers = model.language_model.config.num_hidden_layers + 1
        else:
            num_layers = model.config.num_hidden_layers + 1
    except AttributeError:
        # Fallback
        num_layers = 33 

    print(f"[{model_name}] Calculating hidden states with No-Image Strategy...")

    # Initialize containers
    V_truth_diffs = [[] for _ in range(num_layers)]   # H_gt - H_hall (With Image)
    V_lang_diffs = [[] for _ in range(num_layers)]    # H_unknown_no_img - H_hall_no_img (Language Steering)
    V_prior_diffs = [[] for _ in range(num_layers)]   # H_hall_no_img - H_unknown_no_img (Language Prior direction for projection)
    V_visual_raws = [[] for _ in range(num_layers)]   # H_gt - H_gt_no_img (Raw visual delta)

    DEFAULT_QUESTION = "Describe the image in detail."
    
    for sample in tqdm(dataset, desc=f"Calculating Vectors"):
        try:
            question = sample.get('question', DEFAULT_QUESTION)
            ans_gt = sample['caption']
            ans_hall = sample['h_caption']
            ans_unknown = get_random_unknown_response()
            
            image_path = os.path.join(steering_image_dir, f"{sample['image_id']}.jpg")
            if not os.path.exists(image_path): continue
            
            pil_image = Image.open(image_path).convert('RGB')
            # Local helper to fix model_name context
            def get_states(q, a, img):
                return get_hidden_states_retrospective(model, processor, q, a, img, model_name=model_name)

            # === 1. Image Context States ===
            # H(Image, Factual)
            H_gt = get_states(question, ans_gt, pil_image)
            # H(Image, Hallucination)
            H_hall = get_states(question, ans_hall, pil_image)
            
            # === 2. No-Image Context States ===
            # H(NoImg, Factual) -> Used to extract Visual delta
            H_no_img_gt = get_states(question, ans_gt, None)
            # H(NoImg, Hallucination) -> Hallucination caused by pure language prior
            H_no_img_hall = get_states(question, ans_hall, None)
            # H(NoImg, Unknown) -> Zero point / Neutral point of language
            H_no_img_unknown = get_states(question, ans_unknown, None)

            for i in range(num_layers):
                # A. Truth Vector: Factual vs Hallucination (Standard Steering)
                V_truth_diffs[i].append(H_gt[i] - H_hall[i])
                
                # B. Language Vector: Unknown vs Pure Hallucination (To suppress language-based hallucination)
                V_lang_diffs[i].append(H_no_img_unknown[i] - H_no_img_hall[i])
                
                # C. Prior Vector: Pure Hallucination vs Unknown (The "noise direction" of language prior, subtracted from Visual)
                V_prior_diffs[i].append(H_no_img_hall[i] - H_no_img_unknown[i])
                
                # D. Visual Raw: Image Factual vs No-Image Factual (Raw visual injection)
                V_visual_raws[i].append(H_gt[i] - H_no_img_gt[i])

        except Exception as e:
            print(f"Error processing sample: {e}")
            pass

    # ==========================================================================
    # PCA Calculation and Orthogonalization
    # ==========================================================================
    print(f"[{model_name}] Computing vectors using PCA (with Visual Orthogonalization)...")
    final_steering_vectors = {}

    for i in tqdm(range(num_layers), desc="Processing Layers"):
        if len(V_truth_diffs[i]) < 2: continue
        
        try:
            truth_stack = torch.stack(V_truth_diffs[i])      
            lang_stack = torch.stack(V_lang_diffs[i])        
            prior_stack = torch.stack(V_prior_diffs[i])      
            visual_raw_stack = torch.stack(V_visual_raws[i]) 
            
            # 1. Truth Vector (Direct PCA)
            truth_vec = get_pure_direction_with_pca(truth_stack)

            # 2. Language Vector (Direct PCA) - Used for suppression steering
            lang_vec = get_pure_direction_with_pca(lang_stack)
            
            # 3. Visual Vector (PCA after Orthogonalization)
            # Goal: Extract pure visual information by removing language prior components
            
            # a. Get primary direction of language prior (Masked Hallucination -> Unknown)
            prior_direction = get_pure_direction_with_pca(prior_stack)
            prior_unit = F.normalize(prior_direction, dim=-1) # (D,)

            # b. Calculate projection: Proj = (V_raw · u_prior) * u_prior
            dot_products = visual_raw_stack @ prior_unit.unsqueeze(-1) 
            projections = dot_products * prior_unit.unsqueeze(0) 

            # c. Orthogonalize: Visual_Clean = Visual_Raw - Projection
            visual_ortho_stack = visual_raw_stack - projections

            # d. Perform PCA on cleaned visual samples
            visual_vec = get_pure_direction_with_pca(visual_ortho_stack)

            final_steering_vectors[i] = {
                "truth": truth_vec,
                "language": lang_vec,
                "visual": visual_vec
            }
            
        except Exception as e:
             print(f"Error calculating PCA for layer {i}: {e}")
    
    print(f"[{model_name}] Calculation complete.")
    return final_steering_vectors


def calculate_das_vectors_qwen(model, processor, steering_data_file, steering_image_dir):
    return _calculate_das_vectors_internal(model, processor, steering_data_file, steering_image_dir, "Qwen2.5")


def calculate_das_vectors_qwen3(model, processor, steering_data_file, steering_image_dir):
    return _calculate_das_vectors_internal(model, processor, steering_data_file, steering_image_dir, "Qwen3")

def calculate_das_vectors_llava(model, processor, steering_data_file, steering_image_dir):
    return _calculate_das_vectors_internal(model, processor, steering_data_file, steering_image_dir, "LLaVA")

def calculate_das_vectors_internvl(model, processor, steering_data_file, steering_image_dir):
    return _calculate_das_vectors_internal(model, processor,steering_data_file, steering_image_dir, "InternVL")