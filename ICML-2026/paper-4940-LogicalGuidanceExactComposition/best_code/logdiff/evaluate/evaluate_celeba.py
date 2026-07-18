import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from logdiff.cs_metric import ConformityScorer
from logdiff.datasets.celeba import CelebADataset
from logdiff.evaluate.evaluation_utils import get_null_token, load_models, run_task_evaluation, save_results
from logdiff.score.pipelines import CondDDIMPipeline
from logdiff.score.sampling_compositional import LogicModelWrapper, And, Or_CI, Or_ME, Not
from logdiff.score.sampling_celeba import Hair, Gender
from logdiff.utils import set_seed

import copy
from diffusers import AutoencoderKL
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


import torch
import os
from torchvision.utils import save_image

def get_fid(output_dir, device='cuda'):
    print("Starting FID Evaluation...")
    fid_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.PILToTensor() 
    ])

    real_data = CelebADataset(
        root='data/', 
        split='train', 
        size=128, 
        transforms=fid_transform
    )
    real_loader = DataLoader(real_data, batch_size=128, shuffle=False, num_workers=4)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    print("Computing statistics for Real Images...")
    for batch in real_loader:
        imgs = batch['X'].to(device)
        if imgs.dtype == torch.float:
            imgs = (imgs * 255).byte()
        fid.update(imgs, real=True)

    results = {}
    
    gen_root = os.path.join(output_dir, "gen")
    
    if not os.path.exists(gen_root):
        print(f"Warning: No 'gen' folder found in {output_dir}")
        return results

    class SimpleImageDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            self.files = [f for f in os.listdir(root) if f.endswith(('.png', '.jpg'))]
            self.transform = transform
        def __len__(self): return len(self.files)
        def __getitem__(self, idx):
            from PIL import Image
            img = Image.open(os.path.join(self.root, self.files[idx])).convert("RGB")
            if self.transform: img = self.transform(img)
            return img

    # Loop 1: Methods (e.g., 'logdiff', 'unconditional', 'constant')
    for method_name in os.listdir(gen_root):
        method_path = os.path.join(gen_root, method_name)
        if not os.path.isdir(method_path): continue
        
        results[method_name] = {}
        
        for task_name in os.listdir(method_path):
            task_path = os.path.join(method_path, task_name)
            if not os.path.isdir(task_path): continue
            
            print(f"Calculating FID for Method: [{method_name}] | Task: [{task_name}]")
            
            val_data = SimpleImageDataset(root=task_path, transform=fid_transform)
            if len(val_data) == 0:
                print(f"  No images found in {task_path}")
                continue
                
            val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4)
            
            fid_current = copy.deepcopy(fid) 
            
            for batch in val_loader:
                imgs = batch.to(device)
                if imgs.dtype == torch.float:
                    imgs = (imgs * 255).byte()
                fid_current.update(imgs, real=False)
                
            score = fid_current.compute().item()
            results[method_name][task_name] = score
            print(f"  >> FID: {score:.4f}")
            
    return results


@hydra.main(config_path='../../configs', version_base='1.2', config_name='celeba_inference')
def main(cfg):
    ########## Hyperparameters and settings ##########
    set_seed(cfg.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    logger = logging.getLogger(__name__)
    
    EVAL_TOTAL_SAMPLES = 5000
    BATCH_SIZE = cfg.get("batch_size", 100)
    GUIDANCE = cfg.get("guidance", None)
    USE_NEG_GUIDANCE = cfg.get("use_neg_guidance", False)
    EVAL_BASELINES = cfg.get("evaluate_baselines", True)
    NUM_STEPS = 1000
    output_dir = f"{output_dir}_{cfg.dataset.train_dataset._target_}_{GUIDANCE['atom']}_{cfg.output_suffix}"

    ########## Load model #############
    logger.info("Loading Diffusion Model and Classifiers")
    scheduler = instantiate(cfg.noise_scheduler)
    model, judge_classifier, composition_classifier = load_models(cfg, device)
    vae = AutoencoderKL.from_pretrained('black-forest-labs/FLUX.1-schnell', subfolder='vae', cache_dir='../checkpoints')
    vae.eval()
    vae.to(device)

    ########## Setup Pipeline #############
    expr_wrapper = LogicModelWrapper(model, composition_classifier, USE_NEG_GUIDANCE)
    pipe = CondDDIMPipeline(net=expr_wrapper, scheduler=scheduler, vae=vae)
    null_token = get_null_token(cfg, BATCH_SIZE, device)

    results = {}

    # Task Definitions
    dataset_config = {
        "LOGIC_GROUP_NAMES": ["Gender", "Hair"],
        "ATTRIBUTE_CLASSES": {
            "Gender": Gender,
            "Hair": Hair,
        },
        "ATTRIBUTE_OPTIONS": {
            "Gender": 2,   
            "Hair": 5,  # 5 hair colors: Black (0), Blond (1), Brown (2), Gray (3), Bald (4)
        },
    }

    dataset_config["CLASSES_ATTRIBUTES"] = {v: k for k, v in dataset_config["ATTRIBUTE_CLASSES"].items()}

    def celeba_judge_preprocess(images):
        # Match judge classifier training pipeline: resize to 64x64, normalize to [-1, 1].
        images = images.clamp(0, 1)
        if images.shape[-2:] != (64, 64):
            images = F.interpolate(images, size=(64, 64), mode="bilinear", align_corners=False)
        return images * 2 - 1

    cs = ConformityScorer(
        judge_classifier,
        dataset_config["LOGIC_GROUP_NAMES"],
        dataset_config["CLASSES_ATTRIBUTES"],
        preprocess=celeba_judge_preprocess,
    )

    dataset_config["ATTRIBUTE_OPTIONS"]["Gender"]
    tasks = [
        (
            "AND (Hair, Gender)", 
            lambda: And(Gender(np.random.randint(dataset_config["ATTRIBUTE_OPTIONS"]["Gender"])), 
                        Hair(np.random.randint(dataset_config["ATTRIBUTE_OPTIONS"]["Hair"])))
        ),
        (
            "NOT ",        
            lambda: Not(Gender(np.random.randint(dataset_config["ATTRIBUTE_OPTIONS"]["Gender"]))) 
                    if np.random.rand() < 0.5 
                    else Not(Hair(np.random.randint(dataset_config["ATTRIBUTE_OPTIONS"]["Hair"])))
        ),
        (
            "OR (CI)",        
            lambda: Or_CI(Hair(np.random.randint(dataset_config["ATTRIBUTE_OPTIONS"]["Hair"])), 
                        Gender(np.random.randint(dataset_config["ATTRIBUTE_OPTIONS"]["Gender"])))
        ), 
        (
            "OR (ME)",        
            lambda: Or_ME(
                *map(Hair, np.random.choice(
                    dataset_config["ATTRIBUTE_OPTIONS"]["Hair"], size=2, replace=False,
                ))
            )
        ),

    ]

    # Execution Loop
    for task_name, query_generator in tasks:
        acc = run_task_evaluation(task_name, query_generator, logger, pipe, cs, 
                                  EVAL_TOTAL_SAMPLES, BATCH_SIZE, GUIDANCE, NUM_STEPS, 
                                  null_token, dataset_config["ATTRIBUTE_OPTIONS"], output_dir, EVAL_BASELINES)
        results[task_name] = acc
        save_results(results, EVAL_TOTAL_SAMPLES * BATCH_SIZE, output_dir)

    fid_results = get_fid(output_dir, device)
    save_results(results, EVAL_TOTAL_SAMPLES * BATCH_SIZE, output_dir, fid_results=fid_results)

if __name__ == "__main__":
    main()
