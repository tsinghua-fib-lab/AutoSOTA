import hydra
from hydra.utils import instantiate
from utils import set_seed
import pytorch_lightning as pl
import torch

@hydra.main(config_path='../configs',version_base='1.2',config_name='cmnist')
def main(cfg):
    ########## Hyperparameters and settings ##########
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision('medium')

    ########## Dataset ##########
    train_dataset = instantiate(cfg.dataset.train_dataset)
    val_dataset = instantiate(cfg.dataset.val_dataset)
   
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,**cfg.dataset.train_dataloader)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,**cfg.dataset.val_dataloader)

    ########## Callbacks and Logger ##########
    callbacks,loggers = [],[]
    for callback_name, callback in cfg.callbacks.items():
        callbacks.append(instantiate(callback))
    for logger_name, logger in cfg.loggers.items():
        loggers.append(instantiate(logger))

    ########## Train the diffusion model #############
    diff_process = instantiate(cfg.diffusion, 
                               null_token=cfg.diffusion.model.num_class_per_label)
    diff_trainer = pl.Trainer(callbacks=callbacks, 
                              logger=loggers,
                              **cfg.trainer) 
    diff_trainer.fit(diff_process, train_dataloader, val_dataloader)
    
if __name__ == "__main__":  
    main()

