import os
import yaml
import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import torch

from sae_tools.data_loader import get_adapter
from sae_tools.utils import FilenameConstructor
from sae_tools.model.load_model import (
    load_hooked_transformer_offline,
    load_custom_batch_topk_as_jumprelu
)
from sae_tools.model.run import generate_activations

BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = os.getenv("MODEL_ROOT")
SAE_ROOT = os.getenv("SAE_ROOT")
DATASET_ROOT = os.getenv("DATASET_ROOT")

def load_model(
    model_name: str,
    model_path: str,
    sae_id: str,
    sae_path: str,
    layer: int,
    device: str
):
    model_path = os.path.join(MODEL_ROOT, model_path)
    sae_path = os.path.join(SAE_ROOT, sae_path)

    tokenizer, model = load_hooked_transformer_offline(
        model_name=model_name,
        model_path=model_path,
        device=device
    )

    sae = load_custom_batch_topk_as_jumprelu(
        model_name=model_name,
        sae_id=sae_id,
        sae_path=sae_path,
        layer=layer,
        device=device
    )
    return tokenizer, model, sae

def main(args):
    tokenizer, model, sae = load_model(
        model_name=args.model_name,
        model_path=args.model_path,
        sae_id=args.sae_id,
        sae_path=args.sae_path,
        layer=args.layer,
        device=args.device
    )

    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    max_samples = dataset_config.get('max_samples', -1)
    datasets = dataset_config.get('datasets', [])
    print(f">>> Found {len(datasets)} datasets in {args.dataset_config}")

    filecon = FilenameConstructor(
        args.model_name,
        args.output_dir
    )
    for dataset_info in datasets:
        dataset_name = dataset_info.get('name')
        dataset_type = dataset_info.get('type', 'prompt')
        dataset_folder = dataset_info.get('folder', '')
        dataset_path = os.path.join(DATASET_ROOT, dataset_folder)
        
        adapter = get_adapter(dataset_name)

        dataset = adapter.load(
            dataset_path,
            max_samples,
            split = dataset_info.get('split', None),
            subset = dataset_info.get('subset', None)
        )

        print(">>> Start Inference...")
        print(f">>> Data type: {dataset_type}")
        
        results = generate_activations(
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            layer=args.layer,
            dataset=dataset,
            data_type=dataset_type,
            batch_size=1 # TODO: args.batch_size not used because the padding logic is not implemented
        )
        output_file = filecon.file_name("Guard", dataset_name, "predictions", "pt")
        torch.save(results, output_file)
        print(f">>> Results saved to {output_file}")
    
    print(f"\n>>> All datasets processed successfully!")

DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_MODEL_PATH = "Qwen/Qwen3Guard-Gen-8B"
DEFAULT_SAE_ID = "adamkarvonen/qwen3-8b-saes"
DEFAULT_SAE_PATH = "adamkarvonen/qwen3-8b-saes/saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_18/trainer_2/ae.pt"
DEFAULT_SAE_LAYER = 18
DEFAULT_DATASETS = BASE_DIR / "configs/datasets/datasets_prompt.yaml"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"
DEFAULT_DEVICE = "cuda"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate New Method Model")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    
    parser.add_argument("--sae_id", type=str, default=DEFAULT_SAE_ID)
    parser.add_argument("--sae_path", type=str, default=DEFAULT_SAE_PATH)
    parser.add_argument("--layer", type=int, default=DEFAULT_SAE_LAYER)
    
    parser.add_argument("--dataset_config", type=str, default=DEFAULT_DATASETS)
    
    parser.add_argument("--batch_size", type=int, default=1, help="not used")
    
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    args = parser.parse_args()
    main(args)