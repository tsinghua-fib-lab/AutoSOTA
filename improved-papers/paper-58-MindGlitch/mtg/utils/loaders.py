import sys
from omegaconf import OmegaConf
import hydra
from safetensors.torch import load_file
from segment_anything import build_sam, SamPredictor
from huggingface_hub import hf_hub_download

from configs.config import CLEANDIFT_PATH, CLEANDIFT_CONFIG_PATH, SAM_CHECKPOINT_PATH
from mtg.utils.inpaint import get_inpainting_pipe
from mtg.utils.grounding_sam import (
    load_model_hf,
    ckpt_repo_id,
    ckpt_filenmae,
    ckpt_config_filename,
    sam_checkpoint,
)


def cleandift_loader(DEVICE):
    sys.path.append(CLEANDIFT_PATH)
    cfg_model = OmegaConf.load(CLEANDIFT_CONFIG_PATH)["model"]
    print("Loaded config successfully")
    cfg_model = hydra.utils.instantiate(cfg_model)
    print("Instantiated model successfully")
    model = cfg_model.to(DEVICE).bfloat16()
    ckpt_pth = hf_hub_download(repo_id="CompVis/cleandift", filename="cleandift_sd21_full.safetensors")
    print(f"Downloaded weights file saved at: {ckpt_pth}")
    state_dict = load_file(ckpt_pth)
    model.load_state_dict(state_dict, strict=True)
    model = model.eval()
    return model


def groundingsam_loader(DEVICE):

    ###### Grounding DINO ######
    # Used to segment the main object in the image given the object description from the dataset
    print("==> Loading Grounding DINO model")
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    ###### SAM ######
    # Segments the main object and parts of the object given a BB or a point
    print("==> Loading SAM model")
    sam = build_sam(checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    return groundingdino_model, sam_predictor


def inpainting_model_loader(inpainting_model: str, DEVICE):
    ###### Inpainting ######
    print("==> Loading Inpainting model")
    inpainting_pipe = get_inpainting_pipe(inpainting_model, DEVICE)
    inpainting_pipe.set_progress_bar_config(disable=True)
    return inpainting_pipe
