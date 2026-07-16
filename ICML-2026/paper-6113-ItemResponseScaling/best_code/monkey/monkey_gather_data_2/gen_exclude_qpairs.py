from pathlib import Path
from huggingface_hub import snapshot_download

def list_prompts(model_dir):
    prompts = set()
    for pdir in model_dir.glob("prompt=*"):
        prompts.add(pdir.name)
    return prompts
    
if __name__ == "__main__":
    base_dir = Path(snapshot_download(
        repo_id="stair-lab/denoise_eval",
        repo_type="dataset"
    ))

    scen = "mmlu_pro"
    model1 = "Phi-4-mini-reasoning"
    model2 = "Qwen3-8B"
    dir1 = base_dir / scen / model1
    dir2 = base_dir / scen / model2

    prompts1 = list_prompts(dir1)
    prompts2 = list_prompts(dir2)
    exclude_qpairs = sorted(prompts1 - prompts2)

    print(f"Found {len(exclude_qpairs)} prompts in '{model1}' but missing in '{model2}':")
    for prompt in exclude_qpairs:
        print(f"('{scen}', '{prompt}'),")
