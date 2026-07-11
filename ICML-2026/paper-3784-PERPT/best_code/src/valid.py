import argparse
import json
import os
import re
from evodiff.collaters import OAMaskCollater
from evodiff.losses import OAMaskedCrossEntropyLoss
from evodiff.utils import Tokenizer
import numpy as np
from sequence_models.constants import MSA_ALPHABET
from sequence_models.convolutional import ByteNetLM
from sequence_models.datasets import UniRefDataset
from sequence_models.metrics import MaskedAccuracy
from sequence_models.samplers import SortishSampler, ApproxBatchSampler
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("--data", type=str, default="valid")  # valid, test, or rtest
    parser.add_argument("--max_tokens", type=int, default=5000)  # whatever will fit on gpu mem
    parser.add_argument("--last_eval", type=int, default=5000)  # steps of last checkpoint evaluated
    parser.add_argument("--no_wandb", action="store_true")  # use to disable wandb
    args = parser.parse_args()

    wandb_logging = False if args.no_wandb else True
    validation(args, wandb_logging=wandb_logging)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_model_data(checkpoint_path, tokenizer=Tokenizer()):
    config_path = checkpoint_path + "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config, tokenizer


def initiate_model(config, tokenizer):
    n_tokens = len(MSA_ALPHABET)
    d_embed = config["d_embed"]
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    kernel_size = config["kernel_size"]
    r = config["r"]
    masking_idx = tokenizer.mask_id
    if "slim" in config:
        slim = config["slim"]
    else:
        slim = True
    if "activation" in config:
        activation = config["activation"]
    else:
        activation = "relu"
    dropout = config["dropout"]
    model = ByteNetLM(
        n_tokens,
        d_embed,
        d_model,
        n_layers,
        kernel_size,
        r,
        causal=False,
        padding_idx=masking_idx,
        dropout=dropout,
        tie_weights=False,
        final_ln=True,
        slim=slim,
        activation=activation,
    )
    return model


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def sort_numerical(seq):
    return sorted(seq, key=alphanum_key)


def is_amlt():
    return os.environ.get("AMLT_OUTPUT_DIR", None) is not None


def get_dataloader(config, tokenizer, rank, world_size, v_data="valid", max_tokens=5000):
    if is_amlt():
        data_top_dir = "/mnt/data/data/"
        # data_top_dir = args.data_root or "/ddn/evodiff/"
        data_dir = os.path.join(data_top_dir, "uniref50_202401/")
    else:
        print("FOR DEBUGGING ONLY - using old data")
        data_top_dir = "/home/v-salamdari/Desktop/DMs/data/"
        data_dir = os.path.join(data_top_dir, "uniref50" + "/")  # for debugging only

    collater = OAMaskCollater(tokenizer=tokenizer)

    # load the dataset
    ds_valid = UniRefDataset(data_dir, v_data, structure=False)
    valid_idx = ds_valid.indices
    metadata = np.load(os.path.join(data_dir, "lengths_and_offsets.npz"))
    len_valid = metadata["ells"][valid_idx]
    valid_sortish_sampler = SortishSampler(len_valid, 1000, num_replicas=world_size, rank=rank)
    valid_sampler = ApproxBatchSampler(valid_sortish_sampler, max_tokens, config["max_batch_size"], len_valid)
    dl_valid = DataLoader(dataset=ds_valid, batch_sampler=valid_sampler, num_workers=8, collate_fn=collater)
    if rank == 0:
        print(f"Validating on {len(dl_valid.dataset)} sequences.")
    return dl_valid


def validation(args, wandb_logging=True):
    seed_everything(0)  # use default

    # run validation on 1 gpu
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(f"cuda:{local_rank}")
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    print(f"Starting job on rank {rank} with local rank {local_rank} and world size {world_size}")

    if local_rank == 0 and wandb_logging:
        wandb.init()

    def epoch(model, last_step=0):
        model = model.eval()
        loader = dl_valid

        losses = []
        nll_losses = []
        accus = []
        ns = []
        num_seqs = []
        for i, batch in enumerate(loader):
            new_loss, new_nll_loss, new_accu, new_n, new_seqs, new_processed = step(model, batch)

            losses.append(new_loss.item())
            nll_losses.append(new_nll_loss.item())
            accus.append(new_accu.item())
            ns.append(new_n.item())
            num_seqs.append(new_seqs.item())
        ns = torch.tensor(sum(ns)).to(device)
        losses = torch.tensor(sum(losses)).to(device)
        nll_losses = torch.tensor(sum(nll_losses)).to(device)
        accus = torch.tensor(sum(accus)).to(device)
        num_seqs = torch.tensor(sum(num_seqs)).to(device)
        # reduce to rank 0
        reduce_tensor = torch.stack((ns, losses, nll_losses, accus, num_seqs))
        dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
        ns = reduce_tensor[0]
        losses = reduce_tensor[1]
        nll_losses = reduce_tensor[2]
        accus = reduce_tensor[3]
        num_seqs = reduce_tensor[4]

        if rank == 0:
            print("Last step:", num_seqs, "sequences seen of", len(dl_valid.dataset), "total")
            total_n = ns.item()
            r_loss = losses.item() / total_n
            r_nll_loss = nll_losses.item() / total_n
            raccu = accus.item() / total_n

            if wandb_logging:
                # log metrics to wandb
                wandb.log(
                    {
                        "loss": r_loss,
                        "nll": r_nll_loss,
                        "accu": raccu,
                        "checkpoint_step": last_step,
                    }
                )
        return i

    def step(model, batch):
        src, timestep, tgt, mask = batch
        mask = mask.to(device)
        timestep = timestep.to(device)
        src = src.to(device)
        tgt = tgt.to(device)
        input_mask = (src != padding_idx).float()
        n_tokens = mask.sum()

        n_processed = input_mask.sum()
        n_seqs = torch.tensor(len(src), device=device)
        outputs = model(src, input_mask=input_mask.unsqueeze(-1))
        ce_loss, nll_loss = loss_func(outputs, tgt, mask, timestep, input_mask)  # sum(loss per token)
        loss = ce_loss
        accu = accu_func(outputs, tgt, mask) * n_tokens
        return loss, nll_loss, accu, n_tokens, n_seqs, n_processed

    config, tokenizer = load_model_data(args.checkpoint_path)

    padding_idx = tokenizer.pad_id
    dl_valid = get_dataloader(
        config, tokenizer, rank=rank, world_size=world_size, v_data=args.data, max_tokens=args.max_tokens
    )
    loss_func = OAMaskedCrossEntropyLoss(reweight=True)
    accu_func = MaskedAccuracy()

    # iterate validation over checkpoint files
    checkpoint_files = [filename for filename in os.listdir(args.checkpoint_path) if filename.startswith("checkpoint")]
    checkpoint_files = sort_numerical(checkpoint_files)

    for output in checkpoint_files[::-4]:  # skip every 3rd file, and iterate through checkpoints in reverse
        last_step = int(output.split("checkpoint")[-1].split(".")[0])
        if last_step > args.last_eval:
            # Initiate model
            model = initiate_model(config, tokenizer)
            model = model.to(device)

            # Load checkpoint
            state_dict = os.path.join(args.checkpoint_path, output)
            print(f"device: {device} Loading weights from {state_dict}...")

            sd = torch.load(state_dict, map_location=lambda storage, loc: storage.cuda(local_rank))
            model.load_state_dict(sd["model_state_dict"])
            model = DDP(model, device_ids=[device])
            with torch.no_grad():
                epoch(model, last_step=last_step)
                dl_valid.batch_sampler.sampler.set_epoch(0)


if __name__ == "__main__":
    main()
