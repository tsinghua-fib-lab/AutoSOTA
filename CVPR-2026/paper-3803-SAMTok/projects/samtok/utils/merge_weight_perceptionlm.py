import torch
from mmengine.config import Config, ConfigDict
from transformers import AutoProcessor
from projects.samtok.models import PerceptionLM_TokenMask

if __name__ == "__main__":
    save_path = "./work_dirs/perceptionlm_1b_mt256x2_alldata/perceptionlm_1b_mt_instruct"
    cfg = Config.fromfile('projects/samtok/configs/perceptionlm_1b_mt256x2.py')
   
    model = PerceptionLM_TokenMask(
        mllm=cfg.model.mllm,
        llm_lora=cfg.model.llm_lora,
        freeze_llm=cfg.model.freeze_llm,
        freeze_visual_encoder=cfg.model.freeze_visual_encoder,
        freeze_connector=cfg.model.freeze_connector,
        unfreeze_vocab=cfg.model.unfreeze_vocab,
        unfreeze_lm_head=cfg.model.unfreeze_lm_head,
        use_activation_checkpointing=cfg.model.use_activation_checkpointing,
        pretrained_pth=cfg.model.pretrained_pth,
    )

    processor = AutoProcessor.from_pretrained(cfg.mllm_name_or_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    pth_path = "./work_dirs/perceptionlm_1b_mt256x2_alldata/iter_xxx.pth"
    state = torch.load(pth_path, map_location="cpu", weights_only=False)
    model_sd = (state.get("state_dict")
                or state.get("model")
                or state.get("module")
                or state)
    if any(k.startswith("module.") for k in model_sd.keys()):
        model_sd = {k.replace("module.", "", 1): v for k, v in model_sd.items()}
    
    model.load_state_dict(model_sd, strict=False)
    model.model.tie_weights()

    model.model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Done!")