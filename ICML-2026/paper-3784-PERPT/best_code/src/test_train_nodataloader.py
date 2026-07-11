import argparse
import datetime
import functools
import json
import os
import random
from typing import Optional, Sequence, Tuple, Type


import numpy as np
import wandb

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

from evodiff.utils import Tokenizer

# import gamba using sys.append
import sys

sys.path.append("../gamba")

from gamba.activation_checkpointing import apply_activation_checkpointing
from gamba.collators import gLMCollator, LMCollator, OAMaskCollator
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.datasets import ConservationDataset
from gamba.model import (
    ARDiffusionModel,
    OrderAgnosticDiffusionModel,
    JambagambaModel,
    OTHER_METRICS_KEY,
)
from gamba.model import create_model


import os
import torch


# default values for RANK, LOCAL_RANK, and WORLD_SIZE if not set
ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp") + "/"
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
DEVICE = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")


def is_amlt() -> bool:
    return os.environ.get("AMLT_OUTPUT_DIR", None) is not None


def load_config_and_model(
    config_fpath: str,
) -> Tuple[dict, Tokenizer, nn.Module, Type[nn.Module]]:
    """Parses the experiment config to load the model and tokenizer

    Parameters:
    -----------
    config_fpath: str
        The path to the experiment config file

    Returns:
    --------
    config: dict
        The experiment config
    tokenizer: Tokenizer
        The model's tokenizer
    model: nn.Module
        A task-wrapped version of the specified model, which returns the appropriate loss and metrics
    block: Type[nn.Module]
        The block class used repeatedly in the module. It should not be split by any sharding.
    """
    with open(config_fpath, "r") as f:
        config = json.load(f)
    config["task"] = config["task"].lower().strip()
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    task = TaskType(config["task"].lower().strip())

    print(
        f"Task: {task}, Model: {config['model_type']}, Dataset: {config['dataset']}, Model Config: {config['model_config']}"
    )
    # create the model
    model, block = create_model(
        task, config["model_type"], config["model_config"], tokenizer.mask_id.item()
    )

    # add the task-specific wrapper
    aux_loss_weight = config.get("aux_loss_weight", 0.0)
    if task == TaskType.OADM:
        model = OrderAgnosticDiffusionModel(
            model, tokenizer.pad_id, aux_loss_weight=aux_loss_weight
        )
    elif task == TaskType.LM:
        model = ARDiffusionModel(model, aux_loss_weight=aux_loss_weight)
    elif task == TaskType.GLM:
        model = JambagambaModel(
            model, d_model=576, nhead=8, n_layers=6, padding_id=0, dim_feedfoward=576
        )
    else:
        raise ValueError(f"Unknown task: {config['task']}")
    return config, tokenizer, model, block


def get_dataloader(
    config: dict, tokenizer: Tokenizer, args: argparse.Namespace
) -> DataLoader:
    import os

    # Print the directory of the currently running script
    print(
        "Directory of the running script:", os.path.dirname(os.path.abspath(__file__))
    )
    if is_amlt():
        data_top_dir = args.data_root or "/ddn/evodiff/"
    else:
        data_top_dir = args.data_root or "/home/mica/gamba/data_processing/data/"

    dataset = config["dataset"]
    data_dir = os.path.join(data_top_dir, dataset + "/")

    if config["task"] == "oadm":
        collator = OAMaskCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=config.get("pad_to_multiple_of", None),
        )
    elif config["task"] == "lm":
        collator = LMCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=config.get("pad_to_multiple_of", None),
            flip_prob=config.get("flip_prob", 0.0),
            fim_prob=config.get("fim_prob", 0.0),
            swap_bos_eos_on_flip=config.get("swap_bos_eos_on_flip", True),
        )
    elif config["task"] == "glm":
        collator = gLMCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=config.get("pad_to_multiple_of", None),
        )
    else:
        raise ValueError(f"Unknown task: {config['task']}")

    # create the dataloader
    if args.mini_run:
        # load the dataset
        print("making the dataset")
        ds_train = ConservationDataset(
            data_dir,
            "train",
            num_sequences=16,
            max_len=config["max_len"],
            specific_chromosomes=["2"],
        )
        train_idx = ds_train.indices
        print(f"len(train_idx): {len(train_idx)}")

        # call gLM collator on batch_tensor
        batch_tensor = collator(ds_train)
        print("Batch tensor:", batch_tensor)
        ds_train = batch_tensor



        # dl_train = DataLoader(
        #     dataset=ds_train,
        #     shuffle=True,
        #     batch_size=64,
        #     num_workers=16,
        #     collate_fn=collator,
        # )
        # # load the val dataset:
        # ds_val = ConservationDataset(
        #     data_dir,
        #     "valid",
        #     num_sequences=10,
        #     max_len=config["max_len"],
        #     specific_chromosomes=["12"],
        # )
        # val_idx = ds_val.indices
        # print(f"len(val_idx): {len(val_idx)}")
        # dl_val = DataLoader(
        #     dataset=ds_val,
        #     shuffle=True,
        #     batch_size=64,
        #     num_workers=16,
        #     collate_fn=collator,
        # )
    else:
        # load the dataset
        ds_train = ConservationDataset(
            data_dir, "train", num_sequences=230000, max_len=config["max_len"]
        )
        # load the val dataset
        ds_val = ConservationDataset(
            data_dir, "valid", num_sequences=10000, max_len=config["max_len"]
        )
        # metadata = np.load(os.path.join(data_dir, "lengths_and_offsets.npz"))
        # len_train = np.minimum(metadata["ells"][train_idx], config["max_len"])
        train_idx = ds_train.indices
        len_train = len(train_idx)
        val_idx = ds_val.indices
        len_val = len(val_idx)
        print(f"len(train_idx): {len(train_idx)}")
        train_sortish_sampler = SortishSampler(
            len_train, config["bucket_size"], num_replicas=WORLD_SIZE, rank=RANK
        )
        train_sampler = ApproxBatchSampler(
            train_sortish_sampler,
            config["max_tokens"],
            config["max_batch_size"],
            len_train,
            batch_mult=8,
        )

        dl_train = DataLoader(
            dataset=ds_train,
            batch_sampler=train_sampler,
            num_workers=8,
            collate_fn=collator,
            pin_memory=True,
        )
        dl_val = DataLoader(
            dataset=ds_val,
            batch_size=8,
            num_workers=8,
            collate_fn=collator,
            pin_memory=True,
        )
        if RANK == 0:
            print(f"Validating on {len_val} sequences.")
    return ds_train #dl_train, dl_val


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def step(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    training: bool = True,
) -> dict:
    if any(el.numel() for el in batch) == 0:
        raise ValueError("Empty tensor in batch")

    batch = [el.to(DEVICE) for el in batch]

    if training:
        # step through model
        optimizer.zero_grad()
        print(f"entering model with batch {batch[0].shape}")
        outputs = model(*batch)
        # try clipping the gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        outputs["loss"].backward()
        optimizer.step()
        scheduler.step()
    else:
        # validation
        with torch.no_grad():
            outputs = model(*batch)
    return outputs


def validation(model, val_loader, args, epoch=None, train_step=None, csv_fpath=None):
    # log average val
    # log val ce loss

    if not args.mini_run:
        if args.verbose:
            print(RANK, "Setting epoch")
        val_loader.batch_sampler.sampler.set_epoch(0)

    model = model.eval()

    if args.verbose:
        print("Starting validation...", RANK)

    total_tokens = 0
    total_seqs = 0
    total_ce_loss = 0
    total_gaussian_loss = 0
    for batch in val_loader:
        output = step(model, batch, None, None, training=False)
        with torch.no_grad():
            reduce_tensor = torch.stack(
                (
                    output["n_processed"],
                    output["n_seqs"],
                    output["cross_entropy_loss"],
                    output["gaussian_loss"],
                )
            )
            dist.reduce(reduce_tensor, 0, op=dist.ReduceOp.SUM)
        total_tokens += int(reduce_tensor[0].item())
        total_seqs += int(reduce_tensor[1].item())
        total_ce_loss += reduce_tensor[2].item()
        total_gaussian_loss += reduce_tensor[3].item()
    if RANK == 0:
        with open(csv_fpath, "a") as f:
            f.write(
                f"{epoch},{train_step},{total_tokens},{total_ce_loss},{total_gaussian_loss}\n"
            )
        wandb.log(
            {
                "val_ce_loss": total_ce_loss,
                "val_gaussian_loss": total_gaussian_loss,
                "tokens_validated": total_tokens,
                "nsteps": train_step,
                "epoch": epoch,
                **{k: v.item() for k, v in output[OTHER_METRICS_KEY].items()},
            }
        )


def save_checkpoint(
    out_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    epoch: int,
    tokens: int,
    sequences: int,
) -> None:
    out_path = os.path.join(out_dir, f"dcp_{step}")
    print(f"Saving checkpoint to {out_path}", RANK, flush=True)
    model_state, optim_state = get_state_dict(model, optimizer)
    sd = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optim_state,
    }
    fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(out_path)
    _ = dcp.save(sd, storage_writer=fs_storage_writer)
    if RANK == 0:
        sched_state = scheduler.state_dict()
        sd = {
            "step": step,
            "tokens": tokens,
            "sequences": sequences,
            "scheduler_state_dict": sched_state,
            "epoch": epoch,
        }
        torch.save(sd, os.path.join(out_path, "scheduler.pt"))


def epoch(
    model: nn.Module,
    dataloader: DataLoader,
    #val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    current_epoch: int,
    current_step: int,
    current_tokens: int,
    current_sequences: int,
    out_fpath: str,
) -> Tuple[int, int, int]:
    model = model.train()

    total_steps = current_step
    total_tokens = current_tokens
    total_seq = current_sequences

    csv_fpath = os.path.join(out_fpath, "val.csv")

    # if total_steps == 0:
        # validation(
        #     model,
        #     val_loader,
        #     args,
        #     epoch=current_epoch,
        #     train_step=total_steps,
        #     csv_fpath=csv_fpath,
        # )

    #for batch in dataloader:
    batch = dataloader
    if args.verbose:
        print("rank", RANK, "batchsize", batch[0].shape)

    print("starting one step...")

    # def step(model, batch, device, optimizer, scheduler) -> dict:
    output = step(model, batch, optimizer, scheduler)

    # Accurate metric logging with reduce
    # Log number of sequences and processed tokens in one operation
    with torch.no_grad():
        reduce_tensor = torch.stack(
            (
                output["n_processed"],
                output["n_seqs"],
                output["cross_entropy_loss"],
                output["gaussian_loss"],
            )
        )
        dist.reduce(reduce_tensor, 0, op=dist.ReduceOp.SUM)

    total_steps += 1
    total_tokens += int(reduce_tensor[0].item())
    total_seq += int(reduce_tensor[1].item())

    if RANK == 0:
        # log metrics to wandb
        wandb.log(
            {
                "loss": output["loss"].item(),
                "cross_entropy_loss": output["cross_entropy_loss"].item(),
                "gaussian_loss": output["gaussian_loss"].item(),
                "nsteps": total_steps,
                "epoch": current_epoch,
                "token_trained": total_tokens,
                "sequences_trained": total_seq,
                "lr": optimizer.param_groups[0]["lr"],
                **{k: v.item() for k, v in output[OTHER_METRICS_KEY].items()},
            }
        )

    if total_steps % args.checkpoint_freq == 0:
        save_checkpoint(
            args.out_fpath,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=total_steps,
            epoch=current_epoch,
            tokens=total_tokens,
            sequences=total_seq,
        )
        # validation at checkpoint_freq
        # validation(
        #     model, val_loader, args, current_epoch, total_steps, csv_fpath=csv_fpath
        # )

    return total_steps, total_tokens, total_seq


def get_latest_dcp_checkpoint_path(ckpt_dir: str, last_step: int = -1) -> Optional[str]:
    ckpt_path = None
    if last_step == -1:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        for dir_name in os.listdir(ckpt_dir):
            if "dcp_" in dir_name:
                step = int(dir_name.split("dcp_")[-1])
                if step > last_step:
                    ckpt_path = os.path.join(ckpt_dir, dir_name)
                    last_step = step
    else:
        ckpt_path = os.path.join(ckpt_dir, f"dcp_{last_step}")
    return ckpt_path


def load_checkpoint(
    model, optimizer, scheduler, ckpt_dir: str, last_step: int = -1
) -> Tuple[int, int, int, int]:
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, last_step=last_step)
    if ckpt_path:
        print(f"Loading weights from {ckpt_path}...", flush=True)
        fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(ckpt_path)

        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        state_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }
        dcp.load(
            state_dict=state_dict,
            storage_reader=fs_storage_reader,
        )
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optimizer_state_dict,
        )

        sd = torch.load(
            os.path.join(ckpt_path, "scheduler.pt"), map_location=torch.device("cpu")
        )
        scheduler.load_state_dict(sd["scheduler_state_dict"])

        # sequences must optionally return 0 for backwards compatibility with old checkpoints
        return sd["epoch"] + 1, sd["step"], sd["tokens"], sd.get("sequences", 0)
    else:
        return 0, 0, 0, 0


def train(args: argparse.Namespace) -> None:
    print(
        f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}"
    )
    seed_everything(args.random_seed)

    dist.init_process_group(backend="nccl")
    # get the config, tokenizer, and model
    if args.verbose:
        print("Initializing model...", RANK)
    config, tokenizer, model, blk_types = load_config_and_model(args.config_fpath)
    if RANK == 0:
        if args.no_wandb:
            wandbmode = "disabled"
        else:
            wandbmode = "online"
        wandb.init(config=config, mode=wandbmode)
    out_fpath = ckpt_dir
    csv_fpath = os.path.join(out_fpath, "val.csv")

    if args.verbose:
        print("Done initializing model.", RANK)

    # store the command line args in the config and dump to disk
    config["dtype"] = args.dtype
    config["random_seed"] = args.random_seed
    config["world_size"] = WORLD_SIZE
    if RANK == 0:
        os.makedirs(args.out_fpath, exist_ok=True)
        with open(os.path.join(args.out_fpath, "config.json"), "w") as f:
            json.dump(config, f)

    # training dtype and local device
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    padding_idx = tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
    if RANK == 0:
        print("Using {} as padding index".format(padding_idx))
        print("Using {} as masking index".format(tokenizer.mask_id))
        print(
            f"Model has {sum(p.numel() for p in model.parameters())} trainable parameters."
        )
    if args.verbose:
        print("Initializing data...", RANK)
    print("Initializing data for training...")
    ds_train  = get_dataloader(config, tokenizer, args) #dl_train, dl_valid = get_dataloader(config, tokenizer, args)
    if args.verbose:
        print("Done initializing data.", RANK)
    if RANK == 0:
        print(f"Training on {len(ds_train)} sequences.") #print(f"Training on {len(dl_train.dataset)} sequences.")
    if args.verbose:
        print("Moving and sharding model...", RANK)
    # set the default device
    torch.cuda.set_device(LOCAL_RANK)

    # setup FSDP
    # don't split ByteNetBlock's across devices
    device_mesh = init_device_mesh("cuda", (WORLD_SIZE,))
    wrap_policy = functools.partial(
       transformer_auto_wrap_policy, transformer_layer_cls=blk_types
    )
    #from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
   
    #wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
    mixed_precision = MixedPrecision(param_dtype=dtype, buffer_dtype=dtype)
    shard_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    bwd_prefetch = BackwardPrefetch.BACKWARD_PRE
    model = FSDP(
        model,
        device_id=DEVICE,
        device_mesh=device_mesh,
        auto_wrap_policy=wrap_policy)#,
        #sharding_strategy=shard_strategy,
        #mixed_precision=mixed_precision,
        #backward_prefetch=bwd_prefetch,
    #)

    # create the optimizer and scheduler
    print("creating optimizer and scheduler")
    epochs = config["epochs"]
    lr = config["lr"]
    warmup_steps = config["warmup_steps"]
    optimizer = Adam(
        model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 0.0)
    )
    #lr_func = transformer_lr(warmup_steps)
    lr_func = warmup(warmup_steps)
    scheduler = LambdaLR(optimizer, lr_func)

    # load the state
    # print("loading state")
    # initial_epoch, total_steps, total_tokens, total_seqs = load_checkpoint(
    #     model, optimizer, scheduler, args.out_fpath, args.last_step
    # )
    initial_epoch = 0
    total_steps = 0
    total_tokens = 0
    total_seqs = 0
    # override from config
    optimizer.param_groups[0]["lr"] = config["lr"] * lr_func(total_steps + 1)
    optimizer.param_groups[0]["initial_lr"] = config["lr"]
    scheduler.base_lrs = [config["lr"]]
    act_ckpt = config.get("activation_checkpointing", None)
    if act_ckpt is not None:
        apply_activation_checkpointing(model, blk_types, act_ckpt)

    # train
    for e in range(initial_epoch, epochs):
        start_time = datetime.datetime.now()
        if not args.mini_run:
            if args.verbose:
                print(RANK, "Setting epoch")
            #dl_train.batch_sampler.sampler.set_epoch(e + 1)

        print("going into epoch")
        total_steps, total_tokens, total_seqs = epoch(
            model,
            ds_train, #dl_train,
            #dl_valid,
            optimizer,
            scheduler,
            args,
            current_epoch=e,
            current_step=total_steps,
            current_tokens=total_tokens,
            current_sequences=total_seqs,
            out_fpath=out_fpath,
        )

        save_checkpoint(
            args.out_fpath,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=total_steps,
            epoch=e,
            tokens=total_tokens,
            sequences=total_seqs,
        )
        # validation(
        #     model,
        #     dl_valid,
        #     args,
        #     e,
        #     total_steps,
        #     csv_fpath,
        # )
        print(f"Epoch {e} complete in {datetime.datetime.now() - start_time}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_fpath")
    parser.add_argument(
        "out_fpath",
        type=str,
        nargs="?",
        default=os.getenv("AMLT_OUTPUT_DIR", "/tmp") + "/",
    )
    parser.add_argument("data_root", type=str, nargs="?", default=None)
    parser.add_argument(
        "--mini_run", action="store_true"
    )  # Set to True if running on subset of data
    parser.add_argument("--checkpoint_freq", type=int, default=2000)  # in steps
    parser.add_argument(
        "--random_seed", type=int, default=0
    )  # lambda reweighting term from Austin D3PM
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--last_step", default=-1, type=int)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
