import os

from models.build import build_model, load_best_model, build_norm_module
from utils.parser import parse_args, load_config
from utils.log import init_wandb
from datasets.build import update_cfg_from_dataset
from trainer import build_trainer
from predictor import Predictor
from utils.misc import set_seeds, set_devices
from tta.tafas import build_adapter
from config import get_norm_module_cfg


def main():
    args = parse_args()
    cfg = load_config(args)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    
    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)

    # set wandb logger
    if cfg.WANDB.ENABLE:
        init_wandb(cfg)

    with open(os.path.join(cfg.RESULT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    
    # set random seed
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None

    if cfg.TRAIN.ENABLE:
        # build trainer
        trainer = build_trainer(cfg, model, norm_module=norm_module)
        trainer.train()
        
    if cfg.TTA.ENABLE or cfg.TEST.ENABLE:
        model = load_best_model(cfg, model)
        if cfg.NORM_MODULE.ENABLE:
            norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)
    if cfg.TTA.ENABLE:
        adapter = build_adapter(cfg, model, norm_module=norm_module)
        adapter.adapt()
    if cfg.TEST.ENABLE:
        predictor = Predictor(cfg, model, norm_module=norm_module)
        predictor.predict()


if __name__ == '__main__':
    main()
