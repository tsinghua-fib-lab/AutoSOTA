# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DP-FTRL training, based on paper
"Practical and Private (Deep) Learning without Sampling or Shuffling"
https://arxiv.org/abs/2103.00039.
"""

from absl import app
from absl import flags

import os
import json
import datetime
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import trange
import numpy as np

import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
from opacus import GradSampleModule

from optimizers import FTRLOptimizer
from ftrl_noise import CummuNoiseTorch, CummuNoiseEffTorch, CummuNoiseLR
from nn import get_nn
from data import get_data
import dp_data
import privacy as privacy_lib
import utils
from utils import EasyDict


FLAGS = flags.FLAGS

flags.DEFINE_enum('data', 'mnist', ['mnist', 'cifar10', 'emnist_merge'], '')

# Training algorithm.
# - ftrl_dp: DP-FTRL / DP-FTRLM (tree noise; Opacus used for clipping only)
# - ftrl_dp_matrix: DP-FTRL / DP-FTRLM with L=R prefix-sum (matrix) noise
# - ftrl_nodp: vanilla (non-private) FTRL/FTRLM
# - sgd_amp: DP-SGD with Opacus accounting (subsampling amplification)
# - sgd_noamp: DP-SGD training, but privacy is *reported* without subsampling amplification
flags.DEFINE_enum('algo', 'ftrl_dp', ['ftrl_dp', 'ftrl_dp_matrix', 'ftrl_nodp', 'sgd_amp', 'sgd_noamp'],
                  'Training algorithm. See source for details.')

flags.DEFINE_boolean('dp_ftrl', True, 'If True, train with DP-FTRL. If False, train with vanilla FTRL.')
flags.DEFINE_float('noise_multiplier', 4, 'Ratio of the standard deviation to the clipping norm.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm.')

flags.DEFINE_integer('restart', 1, 'If > 0, restart the DP-FTRL aggregator every this number of epoch(s).')
flags.DEFINE_boolean('effi_noise', True, 'If True, use tree aggregation proposed in https://privacytools.seas.harvard.edu/files/privacytools/files/honaker.pdf.')
flags.DEFINE_boolean('tree_completion', True, 'If true, use the tree completion trick (tree noise only).')

flags.DEFINE_float('momentum', 0, 'Momentum for DP-FTRL.')
flags.DEFINE_float('learning_rate', 4.0, 'Learning rate.')
flags.DEFINE_integer('batch_size', 500, 'Batch size.')
flags.DEFINE_integer('epochs', 5, 'Number of epochs.')
flags.DEFINE_boolean('dp_dataloader', True, 'Use DP randomized batch-size dataloader.')

flags.DEFINE_integer('report_nimg', -1, 'Write to tb every this number of samples. If -1, write every epoch.')


flags.DEFINE_integer('run', 1, '(run-1) will be used for random seed.')
flags.DEFINE_string('dir', '.', 'Directory to write the results.')


def main(argv):
    tf.get_logger().setLevel('ERROR')
    tf.config.experimental.set_visible_devices([], "GPU")

    print(f"DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}") # ADD THIS LINE
    print(f"DEBUG: Detected device = {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}") # ADD THIS LINE
    

    # Setup random seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(FLAGS.run - 1)
    random.seed(FLAGS.run - 1)
    np.random.seed(FLAGS.run - 1)

    # Data
    trainset, testset, ntrain, nclass = get_data(FLAGS.data)
    print('Training set size', trainset.image.shape)



    # Hyperparameters for training.
    epochs = FLAGS.epochs
    batch = FLAGS.batch_size if FLAGS.batch_size > 0 else ntrain
    noise_multiplier = FLAGS.noise_multiplier
    clip = FLAGS.l2_norm_clip

    # Determine target delta (matching the paper's common settings).
    def get_delta(dataset_name: str) -> float:
        if dataset_name in ['mnist', 'cifar10']:
            return 1e-5
        return 1e-6

    delta = get_delta(FLAGS.data)


    # Convert trainset to PyTorch Tensors and create a PyTorch Dataset
    train_images_tensor = torch.from_numpy(trainset.image).float()
    train_labels_tensor = torch.LongTensor(trainset.label)

    pytorch_train_dataset = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)
    pytorch_train_loader = None
    num_batches_fixed = None
    dp_plan = None
    if FLAGS.dp_dataloader:
        dp_plan = dp_data.plan_dp_batches(
            dataset_size=ntrain,
            mean_batch_size=batch,
            delta=delta,
            shuffle=True,
        )
        num_batches_fixed = len(dp_plan["batch_sizes"])
        sampler_eps = dp_plan["sampler_epsilon"]
        dataloader_dp = {
            "enabled": True,
            "epsilon": sampler_eps,
            "delta": delta,
            "mean_batch_size": batch,
        }
    else:
        perm = torch.randperm(ntrain).tolist()
        fixed_dataset = torch.utils.data.Subset(pytorch_train_dataset, perm)

        pytorch_train_loader = torch.utils.data.DataLoader(
            fixed_dataset,
            batch_size=batch,
            shuffle=True,    # DataStream handles shuffling
            drop_last=True
        )
        num_batches_fixed = len(pytorch_train_loader)
        dataloader_dp = {
            "enabled": False,
            "epsilon": None,
            "delta": None,
            "mean_batch_size": batch,
        }

    def _build_dp_loader():
        loader, _, _, num_batches_epoch = dp_data.get_dp_dataloader(
            trainset_images=train_images_tensor,
            trainset_labels=train_labels_tensor,
            ntrain=ntrain,
            mean_batch_size=batch,
            delta=delta,
            shuffle=False,
            plan=dp_plan,
        )
        return loader, num_batches_epoch


    # Backward-compatible behavior: historically --dp_ftrl toggled private vs non-private FTRL.
    # If the user didn't change --algo from its default, respect --dp_ftrl.
    algo = FLAGS.algo
    if algo in ['ftrl_dp', 'ftrl_dp_matrix', 'ftrl_nodp'] and (FLAGS.algo == 'ftrl_dp'):
        if not FLAGS.dp_ftrl:
            algo = 'ftrl_nodp'
    dp_ftrl = algo in ['ftrl_dp', 'ftrl_dp_matrix']
    dp_sgd = algo in ['sgd_amp', 'sgd_noamp']

    restart_epochs = FLAGS.restart
    if algo == 'ftrl_dp_matrix' and FLAGS.dp_dataloader and restart_epochs != 1:
        print("WARNING: ftrl_dp_matrix with randomized batch sizes forces --restart=1 for valid accounting.")
        restart_epochs = 1
    if not restart_epochs:
        FLAGS.tree_completion = False
    if algo == 'ftrl_dp_matrix' and FLAGS.tree_completion:
        print("WARNING: --tree_completion is ignored for ftrl_dp_matrix.")
        FLAGS.tree_completion = False
    effi_noise = FLAGS.effi_noise if algo == 'ftrl_dp' else False
    lr = FLAGS.learning_rate

    report_nimg = ntrain if FLAGS.report_nimg == -1 else FLAGS.report_nimg
    #assert report_nimg % batch == 0

    # Get the name of the output directory.

    dpdl_tag = 'dp_dataloader' if FLAGS.dp_dataloader else 'non_dp_dataloader'
    log_dir = os.path.join(
            FLAGS.dir,
            FLAGS.data,
            dpdl_tag,
            utils.get_fn(
                        EasyDict(batch=batch),
                        EasyDict(
                            dpsgd=(dp_ftrl or dp_sgd),
                            algo=algo,
                            restart=FLAGS.restart,
                            completion=FLAGS.tree_completion,
                            noise=noise_multiplier,
                            clip=clip,
                            mb=1
                        ),
                        [
                            EasyDict({'lr': lr}),
                            EasyDict(
                                m=FLAGS.momentum if FLAGS.momentum > 0 else None,
                                effi=FLAGS.effi_noise
                            ),
                            EasyDict(sd=FLAGS.run)
                        ]
                    )
                )
    print('Model dir', log_dir)
    os.makedirs(log_dir, exist_ok=True)

    def _append_jsonl(path: str, obj: dict) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, sort_keys=True) + "\n")

    def _write_json(path: str, obj: dict) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    # Class to output batches of data
    class DataStream:
        def __init__(self, loader=None, num_batches=None):
            self.loader = loader
            self.num_batches = num_batches
            self._iter = None
            self.shuffle()

        def set_loader(self, loader):
            self.loader = loader
            self._iter = iter(self.loader)

        def set_num_batches(self, num_batches):
            self.num_batches = num_batches

        def shuffle(self):
            if self.loader is not None:
                self._iter = iter(self.loader)
                return
            self.perm = np.random.permutation(ntrain)
            self.i = 0

        def reset(self):
            if self.loader is not None:
                self._iter = iter(self.loader)
                return
            self.i=0

        def __call__(self):
            if self.loader is not None:
                try:
                    data, target = next(self._iter)
                except StopIteration:
                    raise RuntimeError("DP dataloader exhausted before expected steps")
                return data, target
            if self.num_batches is None:
                raise RuntimeError("num_batches is not set for non-DP dataloader")
            if self.i == self.num_batches:
                self.i = 0
            batch_idx = self.perm[self.i * batch:(self.i + 1) * batch]
            self.i += 1
            return trainset.image[batch_idx], trainset.label[batch_idx]

    data_stream = DataStream(pytorch_train_loader, num_batches_fixed)

    def _clip_and_add_noise(model, clip_value: float, noise_multiplier: float,
                            mean_batch_size: int, actual_batch_size: int,
                            device, add_noise: bool) -> None:
        grad_sq = None
        for p in model.parameters():
            if not hasattr(p, "grad_sample"):
                continue
            gs = p.grad_sample
            gs = gs.view(gs.shape[0], -1)
            sq = (gs * gs).sum(dim=1)
            grad_sq = sq if grad_sq is None else (grad_sq + sq)
        if grad_sq is None:
            return
        per_sample_norms = grad_sq.sqrt()
        clip_coef = (clip_value / (per_sample_norms + 1e-6)).clamp(max=1.0)
        for p in model.parameters():
            if not hasattr(p, "grad_sample"):
                continue
            gs = p.grad_sample
            coef = clip_coef.view(-1, *([1] * (gs.dim() - 1)))
            summed = (gs * coef).sum(dim=0)
            p.grad = summed / float(actual_batch_size)
            if add_noise and noise_multiplier > 0:
                noise_std = noise_multiplier * clip_value / float(mean_batch_size)
                p.grad += torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=p.grad.shape,
                    device=p.grad.device,
                )
            p.grad_sample = None

    grad_sample_checked = False

    # Function to conduct training for one epoch
    def train_loop(model, device, optimizer, cumm_noise, epoch, writer, algo_name: str,
                   num_batches_epoch: int, global_step: int, seen_images: int,
                   next_report: int):
        model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total_loss = 0.0
        epoch_seen = 0
        # Use simple range instead of trange to avoid log file clutter
        loop = range(num_batches_epoch)
        print(f'Starting Epoch {epoch+1}/{epochs}...')
        for _ in loop:
            global_step += 1
            data, target = data_stream()
            data = torch.as_tensor(data, dtype=torch.float32, device=device)
            target = torch.as_tensor(target, dtype=torch.long).to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            nonlocal grad_sample_checked
            if not grad_sample_checked:
                missing = [name for name, p in model.named_parameters()
                           if p.requires_grad and not hasattr(p, "grad_sample")]
                if missing:
                    raise RuntimeError(f"Missing grad_sample for parameters: {missing}")
                grad_sample_checked = True

            dp_training = algo_name in ['ftrl_dp', 'ftrl_dp_matrix', 'sgd_amp', 'sgd_noamp']
            clip_value = clip if dp_training else float('inf')
            # --- REPLACEMENT BLOCK START ---
            if algo_name in ['sgd_amp', 'sgd_noamp']:
                # Standard DP-SGD (Opacus handles everything)
                _clip_and_add_noise(
                    model=model,
                    clip_value=clip_value,
                    noise_multiplier=noise_multiplier,
                    mean_batch_size=batch,
                    actual_batch_size=data.shape[0],
                    device=device,
                    add_noise=True,
                )
                optimizer.step()
            else:
                # DP-FTRL: We need to bypass Opacus's step to pass our tuple arguments
                
                # 1. Attempt to trigger Opacus privacy logic (Clipping/Accounting)
                #    If these methods don't exist, Opacus might be doing it via hooks (older versions)
                if hasattr(optimizer, 'original_optimizer'): # This will be True for ftrl_dp / ftrl_dp_matrix
                 # Call the original FTRLOptimizer's step method
                    _clip_and_add_noise(
                        model=model,
                        clip_value=clip_value,
                        noise_multiplier=noise_multiplier,
                        mean_batch_size=batch,
                        actual_batch_size=data.shape[0],
                        device=device,
                        add_noise=False,
                    )
                    optimizer.original_optimizer.step((lr, cumm_noise()))
                else:
                    # This branch should only be hit for ftrl_nodp (non-DP FTRL)
                    # where Opacus is NOT attached, so 'optimizer' is directly FTRLOptimizer
                    _clip_and_add_noise(
                        model=model,
                        clip_value=clip_value,
                        noise_multiplier=noise_multiplier,
                        mean_batch_size=batch,
                        actual_batch_size=data.shape[0],
                        device=device,
                        add_noise=False,
                    )
                    optimizer.step((lr, cumm_noise()))

            total_loss += loss.item()
            epoch_seen += data.shape[0]

            # Check if we just finished an epoch (or passed the report interval)
            seen_images += data.shape[0]
            if seen_images >= next_report:
                acc_train, acc_test = test(model, device)
                writer.add_scalar('eval/accuracy_test', 100 * acc_test, global_step)
                writer.add_scalar('eval/accuracy_train', 100 * acc_train, global_step)
                model.train()
                print('Step %04d Accuracy %.2f' % (global_step, 100 * acc_test))
                next_report += report_nimg

        avg_loss = total_loss / max(1, epoch_seen)
        writer.add_scalar('eval/loss_train', avg_loss, epoch + 1)
        print('Epoch %04d Loss %.2f' % (epoch + 1, avg_loss))
        return global_step, seen_images, next_report


    # Function for evaluating the model to get training and test accuracies
    def test(model, device, desc='Evaluating'):
        model.eval()
        b = 1000
        with torch.no_grad():
            accs = [0, 0]
            for i, dataset in enumerate([trainset, testset]):
                for it in trange(0, dataset.image.shape[0], b, leave=False, desc=desc):
                    data, target = dataset.image[it: it + b], dataset.label[it: it + b]
                    data, target = torch.Tensor(data).to(device), torch.LongTensor(target).to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    accs[i] += pred.eq(target.view_as(pred)).sum().item()
                accs[i] /= dataset.image.shape[0]
        return accs

    # Get model for different dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_nn({'mnist': 'small_nn',
                    'emnist_merge': 'small_nn',
                    'cifar10': 'vgg128'}[FLAGS.data],
                   nclass=nclass).to(device)
    

    # Optimizer + privacy engine setup.
    #
    # For DP-FTRL:
    #   - Use Opacus for per-sample gradient computation + clipping only (noise_multiplier=0).
    #   - Inject correlated tree/matrix noise via CummuNoise* and update via FTRLOptimizer.
    #
    # For DP-SGD:
    #   - Use standard SGD, and let Opacus add i.i.d. Gaussian noise (noise_multiplier>0).
    #   - We keep the same model, batch size, clipping norm, learning rate, and number of steps
    #     so results are comparable to DP-FTRL.
    privacy_engine = None
    shapes = [p.shape for p in model.parameters()]


    model = GradSampleModule(model, loss_reduction="sum")

    for p in model.parameters():
        if not hasattr(p, "grad_sample"):
            p.grad_sample = None

    
    if algo in ['ftrl_dp', 'ftrl_dp_matrix', 'ftrl_nodp']:
        optimizer = FTRLOptimizer(
            model.parameters(),
            momentum=FLAGS.momentum,
            record_last_noise=(algo != 'ftrl_dp_matrix') and restart_epochs > 0,
        )
        if not dp_ftrl:
            # For ftrl_nodp, privacy_engine remains None, as before
            privacy_engine = None # Ensure it's explicitly None if not dp_ftrl
            print("DEBUG: FTRL_NoDP, PrivacyEngine not initialized.")
            
        def get_cumm_noise(algo_name: str, effi_noise: bool, num_steps=None):
            # When DP is disabled (or noise_multiplier=0), return correctly-shaped zeros so
            # downstream optimizer code never relies on broadcasting.
            if (not dp_ftrl) or noise_multiplier == 0:
                return CummuNoiseTorch(0, shapes, device)
            if algo_name == 'ftrl_dp_matrix':
                return CummuNoiseLR(noise_multiplier * clip / batch, shapes, device, num_steps=num_steps)
            if not effi_noise:
                return CummuNoiseTorch(noise_multiplier * clip / batch, shapes, device)
            return CummuNoiseEffTorch(noise_multiplier * clip / batch, shapes, device)

        def _segment_steps(start_epoch, steps_per_epoch):
            if steps_per_epoch is None:
                return None
            if restart_epochs and restart_epochs > 0:
                seg_epochs = min(restart_epochs, epochs - start_epoch)
            else:
                seg_epochs = epochs - start_epoch
            return seg_epochs * steps_per_epoch

        initial_steps = _segment_steps(0, num_batches_fixed) if algo == 'ftrl_dp_matrix' else None
        cumm_noise = get_cumm_noise(algo, effi_noise, initial_steps)

    else: # This block handles 'sgd_amp' and 'sgd_noamp'
        # Standard PyTorch SGD optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=FLAGS.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        cumm_noise = lambda: [torch.zeros(s, device=device) for s in shapes]

    # The training loop.
    writer = SummaryWriter(os.path.join(log_dir, 'tb'))
    global_step = 0
    seen_images = 0
    next_report = report_nimg
    batches_per_epoch = []
    for epoch in range(epochs):
        if FLAGS.dp_dataloader:
            pytorch_train_loader, num_batches_epoch = _build_dp_loader()
            data_stream.set_loader(pytorch_train_loader)
        else:
            num_batches_epoch = num_batches_fixed
        if algo in ['sgd_amp', 'sgd_noamp']:
            # DP-SGD baselines use per-epoch reshuffling.
            data_stream.reset()
        global_step, seen_images, next_report = train_loop(
            model, device, optimizer, cumm_noise, epoch, writer, algo,
            num_batches_epoch, global_step, seen_images, next_report,
        )
        batches_per_epoch.append(num_batches_epoch)
        if algo in ['sgd_amp', 'sgd_noamp']:
            scheduler.step()

        if epoch + 1 == epochs:
            break
        restart_now = (algo in ['ftrl_dp', 'ftrl_dp_matrix', 'ftrl_nodp']) and epoch < epochs - 1 and restart_epochs > 0 and (epoch + 1) % restart_epochs == 0
        if restart_now:
            last_noise = None
            if FLAGS.tree_completion:
                actual_steps = sum(batches_per_epoch[-restart_epochs:])
                next_pow_2 = 2**(actual_steps - 1).bit_length()
                if next_pow_2 > actual_steps:
                    last_noise = cumm_noise.proceed_until(next_pow_2)
            if algo == 'ftrl_dp_matrix':
                last_noise = [torch.zeros(shape, device=device) for shape in shapes]
            
            # Unwrap to find the restart method
            if hasattr(optimizer, 'original_optimizer'):
                optimizer.original_optimizer.restart(last_noise)
            elif hasattr(optimizer, 'optimizer'):
                 optimizer.optimizer.restart(last_noise)
            else:
                optimizer.restart(last_noise)
            next_steps = _segment_steps(epoch + 1, num_batches_fixed) if algo == 'ftrl_dp_matrix' else None
            cumm_noise = get_cumm_noise(algo, effi_noise, next_steps)
            data_stream.shuffle()  # shuffle the data only when restart

    # Report privacy at the end.
    total_steps = int(sum(batches_per_epoch))
    sample_rate=1.0*batch/ntrain
    eps = None
    if algo == 'ftrl_dp_matrix':
        if not batches_per_epoch:
            raise RuntimeError("No batch counts recorded for matrix accounting.")
        steps_per_epoch = batches_per_epoch[0]
        steps_fixed = all(b == steps_per_epoch for b in batches_per_epoch)
        if steps_fixed:
            if restart_epochs and restart_epochs > 0:
                epochs_between_restarts = []
                for start in range(0, epochs, restart_epochs):
                    epochs_between_restarts.append(min(restart_epochs, epochs - start))
            else:
                epochs_between_restarts = [epochs]
            eps = privacy_lib.compute_epsilon_lr(
                steps_per_epoch=steps_per_epoch,
                epochs_between_restarts=epochs_between_restarts,
                noise=noise_multiplier,
                delta=delta,
                verbose=False,
            )
            print(f'Privacy (DP-FTRL matrix): (ε={eps:.3f}, δ={delta}) over {total_steps} steps')
        else:
            if restart_epochs and restart_epochs > 0:
                steps_between_restarts = []
                for start in range(0, epochs, restart_epochs):
                    steps_between_restarts.append(sum(batches_per_epoch[start:start + restart_epochs]))
            else:
                steps_between_restarts = [sum(batches_per_epoch)]
            eps = privacy_lib.compute_epsilon_lr_variable_steps(
                steps_between_restarts=steps_between_restarts,
                noise=noise_multiplier,
                delta=delta,
                verbose=False,
            )
            print(
                "WARNING: Matrix accounting is using variable-steps bound "
                "(fixed-order k=1) because steps per epoch vary."
            )
            print(f'Privacy (DP-FTRL matrix, variable steps): (ε={eps:.3f}, δ={delta}) over {total_steps} steps')
    elif algo == 'ftrl_dp':
        # Translate the restart schedule into the format expected by privacy_lib.
        if restart_epochs and restart_epochs > 0:
            steps_between_restarts = []
            for start in range(0, epochs, restart_epochs):
                steps_between_restarts.append(sum(batches_per_epoch[start:start + restart_epochs]))
        else:
            steps_between_restarts = [sum(batches_per_epoch)]
        eps = privacy_lib.compute_epsilon_tree_variable_steps(
            steps_between_restarts=steps_between_restarts,
            noise=noise_multiplier,
            delta=delta,
            tree_completion=FLAGS.tree_completion,
            verbose=False,
        )
        print(f'Privacy (DP-FTRL): (ε={eps:.3f}, δ={delta}) over {total_steps} steps')
    elif algo == 'sgd_amp':
        # Now get_epsilon is on the accountant object
        eps = privacy_lib.compute_epsilon_sgd_amp(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,  # Use the globally calculated sample_rate
            steps=total_steps,        # Use the globally calculated total_steps
            delta=delta,
            # alphas=None,            # Optional: will use RDPAccountant.DEFAULT_ALPHAS by default
        )
        print(f'Privacy (DP-SGD amp / Custom Accountant): (ε={eps:.3f}, δ={delta}) over {total_steps} steps')
    elif algo == 'sgd_noamp':
        # No-amplification: treat each step as a Gaussian mechanism and compose without subsampling.
        effective_sigma = noise_multiplier / np.sqrt(total_steps)
        eps = privacy_lib.convert_gaussian_renyi_to_dp(effective_sigma, delta, verbose=False)
        print(f'Privacy (DP-SGD no-amp): (ε={eps:.3f}, δ={delta}) over {total_steps} steps')


    dploader_eps=(0 if dataloader_dp["epsilon"] is None else float(dataloader_dp["epsilon"]))
    dploader_delta=(0 if dataloader_dp["delta"] is None else float(dataloader_dp["delta"]))
    print(f"dp_loader privacy: (ε={dploader_eps:.3f}, δ={dploader_delta})")
    # Final evaluation + persistent recording.
    acc_train, acc_test = test(model, device, desc='Final evaluation')
    print(f'Final Accuracy: train={100*acc_train:.2f}%, test={100*acc_test:.2f}%')

    num_batches_mean = int(round(total_steps / max(1, epochs)))
    record = {
        "timestamp": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
        "algo": algo,
        "dataset": FLAGS.data,
        "run": FLAGS.run,
        "ntrain": int(ntrain),
        "ntest": int(testset.image.shape[0]),
        "batch_size": int(batch),
        "epochs": int(epochs),
        "num_batches": int(num_batches_mean),
        "num_batches_per_epoch": [int(x) for x in batches_per_epoch],
        "total_steps": int(total_steps),
        "learning_rate": float(lr),
        "momentum": float(FLAGS.momentum),
        "l2_norm_clip": float(clip),
        "noise_multiplier": float(noise_multiplier),
        "restart": int(restart_epochs),
        "tree_completion": bool(FLAGS.tree_completion),
        "effi_noise": bool(effi_noise),
        "ftrl_noise": (
            "matrix"
            if algo == "ftrl_dp_matrix"
            else ("tree_effi" if algo == "ftrl_dp" and effi_noise else ("tree" if algo == "ftrl_dp" else "na"))
        ),
        "delta": float(delta),
        "epsilon": (None if eps is None else float(eps)),
        "training_dp_delta": float(delta),
        "training_dp_epsilon": (None if eps is None else float(eps)),
        "dataloader_dp_enabled": bool(dataloader_dp["enabled"]),
        "dataloader_dp_epsilon": (None if dataloader_dp["epsilon"] is None else float(dataloader_dp["epsilon"])),
        "dataloader_dp_delta": (None if dataloader_dp["delta"] is None else float(dataloader_dp["delta"])),
        "dataloader_dp_mean_batch_size": int(dataloader_dp["mean_batch_size"]),
        "accuracy_train": float(acc_train),
        "accuracy_test": float(acc_test),
        "log_dir": log_dir,
    }

    # Save a per-run summary into the run directory.
    _write_json(os.path.join(log_dir, 'summary.json'), record)
    # Append into a dataset-level and root-level JSONL for easy sweeping/plotting.
    _append_jsonl(os.path.join(FLAGS.dir, FLAGS.data, 'results.jsonl'), record)
    _append_jsonl(os.path.join(FLAGS.dir, 'results.jsonl'), record)

    writer.close()


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
