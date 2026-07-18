import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import hydra
from hydra.utils import instantiate
from utils import set_seed
import pytorch_lightning as pl
import torch


@hydra.main(config_path='../../configs', config_name='cs_cmnist',version_base='1.2')
def main(cfg):
    ########## Hyperparameters and settings ##########
    set_seed(cfg.seed)

    ########## Dataset ##########    
    train_dataset = instantiate(cfg.dataset.train_dataset)
    val_dataset = instantiate(cfg.dataset.val_dataset)
   
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,**cfg.dataset.train_dataloader)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,**cfg.dataset.val_dataloader)

    ########## Callbacks and Logger ##########
    callbacks = []
    for callback_name, callback in cfg.callbacks.items():
        callbacks.append(instantiate(callback))
    logger = instantiate(cfg.logger)

    classifier_model = instantiate(cfg.trainer)
    classifier_trainer = pl.Trainer(callbacks=callbacks, logger=logger, max_epochs=cfg.trainer.epochs, **cfg.training_config) 
    classifier_trainer.fit(classifier_model, train_dataloader, val_dataloader)

    
if __name__ == "__main__":  
    main()