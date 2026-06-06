
from typing import Iterable
import torch
import time
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                     epoch: int, args = None, logger = None, writer=None):
    model.train()
    criterion.train()

    pbar = tqdm(total=len(data_loader.dataloader), desc=f"Initial Loss Fused: Pending")
    for i, data in enumerate(data_loader):
        samples = data['image'].to(torch.device(args.device))
        targets = data['label'].to(torch.device(args.device))

        output = model(samples)
        loss_final = criterion(output, targets.float())
        cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))

        loss_final_str = '{:.4f}'.format(loss_final.item())
        l = optimizer.param_groups[0]['lr']
        logger.info(f"time -> {cur_time} | Epoch -> {epoch} | image_num -> {data['A_paths']} | loss final -> {loss_final_str} | lr -> {l}")

        pbar.set_description(f"Loss: {loss_final.item():.4f}")
        pbar.update(1)
        optimizer.zero_grad()
        loss_final.backward()
        optimizer.step()

        # global_step = epoch * len(data_loader.dataloader) + i
        # writer.add_scalar("Loss/train", loss_final.item(), global_step)

    pbar.close()


