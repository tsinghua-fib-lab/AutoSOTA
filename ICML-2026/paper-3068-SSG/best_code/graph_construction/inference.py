import os
import sys
import time
import warnings

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_root, 'training', 'imagenet'))
sys.path.insert(0, _root)

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from capture_feature import instrument_forward_and_capture, ExitToMainException
import models as models
import types
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pickle


class ImageFolderWithPath(torchvision.datasets.ImageFolder):
    """Wrap ImageFolder so __getitem__ returns (image, target, path)."""
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, target, path

from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import torch
import tqdm
from torch import nn

SAVE_EVERY_N_SAMPLES=10000

# ========================================
# Asynchronous save helper; must be defined at module level.
# ========================================
def _save_pickle_async(data, filepath):
    """
    Standalone function used with ThreadPoolExecutor.submit.
    Save data asynchronously as a .pkl file.
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        raise

def evaluate(model, criterion, data_loader, device, args, print_freq=1, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    module = model.module if hasattr(model, 'module') else model

    # === Hook: capture the input to the fc layer. ===
    fc_input_buffer = []  # Store the fc input for each batch.

    def fc_pre_hook(m, input):
        # input is a tuple; take the first tensor, detach it, and move it to CPU.
        features = input[0].detach().cpu()
        fc_input_buffer.append(features)

    if not hasattr(module, 'fc'):
        raise ValueError(f"Model has no attribute 'fc'. Available attrs: {dir(module)}")
    
    hook_handle = module.fc.register_forward_pre_hook(fc_pre_hook)

    # === Distributed setup. ===
    rank = torch.distributed.get_rank() if args.distributed else 0
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # === Asynchronous save configuration. ===
    executor = ThreadPoolExecutor(max_workers=1)
    pending_tasks = []
    save_counter = 0
    samples_since_last_save = 0

    # === Class mapping. ===
    dataset = data_loader.dataset
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    try:
        weights_enum = models.__dict__[args.model].weights
        weights = weights_enum.default() if callable(weights_enum) else weights_enum
        categories = weights.meta.get("categories", None)
    except Exception as e:
        print(f"Warning: Could not load category names: {e}")
        categories = None

    # === Per-sample loss function.===
    per_sample_loss_fn = nn.CrossEntropyLoss(reduction='none')

    # === Buffer used for batched saving. ===
    buffer = []

    with torch.inference_mode():
        for batch_idx, (image, target, image_paths) in enumerate(tqdm.tqdm(data_loader, desc=f"Rank {rank}", leave=False)):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Forward pass.
            output = model(image)
            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]

            # Move results to CPU.
            output_cpu = output.cpu().float().numpy()
            per_sample_loss = per_sample_loss_fn(output, target).cpu().numpy()
            pred_classes = output_cpu.argmax(axis=1)
            true_classes = target.cpu().numpy()

            # Get the fc input for the current batch.
            if len(fc_input_buffer) == 0:
                raise RuntimeError("No fc input captured. Check if hook is registered and model runs.")
            fc_features = fc_input_buffer.pop()  # Pop the most recent batch of features.
            if len(fc_features) != batch_size:
                fc_features = fc_features[:batch_size]  # Truncate for alignment in case of sampler mismatch.
            fc_features_np = fc_features.numpy()

            # Build each record.
            for i in range(batch_size):
                pred_cls_id = int(pred_classes[i])
                true_cls_id = int(true_classes[i])

                row = {
                    'image_name': image_paths[i],
                    'logits': output_cpu[i].tolist(),
                    'loss': float(per_sample_loss[i]),
                    'feature': fc_features_np[i].tolist(),  # fc input used as feature
                    'true_class': true_cls_id,
                    'pred_class': pred_cls_id,
                    'correct': pred_cls_id == true_cls_id,
                }

                # Add class names when available.
                if categories and 0 <= true_cls_id < len(categories):
                    row['true_classname'] = categories[true_cls_id]
                else:
                    row['true_classname'] = idx_to_class.get(true_cls_id, "unknown")

                if categories and 0 <= pred_cls_id < len(categories):
                    row['pred_classname'] = categories[pred_cls_id]
                else:
                    row['pred_classname'] = idx_to_class.get(pred_cls_id, "unknown")

                buffer.append(row)

            # Update metrics.
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

            samples_since_last_save += batch_size

            # === Check whether a save is needed. ===
            if samples_since_last_save >= SAVE_EVERY_N_SAMPLES:
                # Submit asynchronous save.
                output_file = os.path.join(output_dir, f"results_rank{rank}_chunk{save_counter:04d}.pkl")
                data_to_save = buffer.copy()
                future = executor.submit(_save_pickle_async, data_to_save, output_file)
                pending_tasks.append(future)

                # Log message.
                print(f"Rank {rank} submitted async save of {len(data_to_save)} samples to {output_file}")

                # Clear the buffer.
                buffer.clear()
                samples_since_last_save = 0
                save_counter += 1

    # === Save any remaining data. ===
    if len(buffer) > 0:
        output_file = os.path.join(output_dir, f"results_rank{rank}_chunk{save_counter:04d}.pkl")
        data_to_save = buffer.copy()
        future = executor.submit(_save_pickle_async, data_to_save, output_file)
        pending_tasks.append(future)
        print(f"Rank {rank} submitted FINAL save of {len(data_to_save)} samples to {output_file}")
        save_counter += 1

    # === Wait for all save tasks to finish. ===
    for future in pending_tasks:
        future.result()  # block until completion
    executor.shutdown(wait=True)

    # === Write a completion marker on rank 0 only.===
    if rank == 0:
        done_file = os.path.join(output_dir, "eval_done.txt")
        with open(done_file, "w") as f:
            f.write(f"Evaluation completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"[Rank 0] Evaluation complete. Marker saved: {done_file}")

    # === Remove the hook. ===
    hook_handle.remove()

    # === Synchronize metrics. ===
    num_processed_samples = utils.reduce_across_processes(sum(len(s) for s in [buffer]))  # approximate
    metric_logger.synchronize_between_processes()
    acc1_avg = metric_logger.acc1.global_avg
    print(f"{header} Acc@1 {acc1_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")

    return acc1_avg



def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def collate_with_path(batch):
    images = []
    targets = []
    paths = []
    for item in batch:
        image, target, path = item
        images.append(image)
        targets.append(target)
        paths.append(path)
    return torch.stack(images), torch.tensor(targets), paths


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    cache_path = _get_cache_path(traindir)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)

    if args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        preprocessing = weights.transforms(antialias=True)
        if args.backend == "tensor":
            preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])
    else:
        preprocessing = presets.ClassificationPresetEval(
            crop_size=val_crop_size,
            resize_size=val_resize_size,
            interpolation=interpolation,
            backend=args.backend,
            use_v2=args.use_v2,
        )

    dataset_test = ImageFolderWithPath(traindir, preprocessing)
    # dataset_test = ImageFolderWithPath(valdir, preprocessing)

    # Keep the original __getitem__ and replace it with a path-returning version.
    original_getitem = dataset_test.__getitem__

    def _new_getitem(index):
        img, target = original_getitem(index)
        path, _ = dataset_test.samples[index]
        return img, target, path

    dataset_test.__getitem__ = _new_getitem

    if args.cache_dataset:
        print(f"Saving dataset_test to {cache_path}")
        utils.mkdir(os.path.dirname(cache_path))
        utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return None, dataset_test, None, test_sampler


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = 1000

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True, collate_fn=collate_with_path
    )

    print("Creating model")
    model = models.__dict__[args.model](weights=args.weights, num_classes=num_classes)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if model_ema:
        evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", args=args)
    else:
        evaluate(model, criterion, data_loader_test, device=device, args=args)
    return


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", required=True, type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--bias-weight-decay", default=None, type=float, help="weight decay for bias parameters of all layers (default: None, same value as --wd)")
    parser.add_argument("--transformer-embedding-decay", default=None, type=float, help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./output", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true")
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)")
    parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument("--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
