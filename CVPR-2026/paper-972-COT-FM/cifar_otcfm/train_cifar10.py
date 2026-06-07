import copy
import os
import wandb
from tqdm import tqdm

import torch
from absl import app
from torchvision import datasets, transforms
from tqdm import trange
from utils_cifar import ema, generate_samples
from torchvision.utils import make_grid

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
from compute_fid_utils import get_fid
from torchcfm.optimal_transport import OTPlanSampler
from ppo_utils import load_ppo_resources, PPOConfig, sample_noise
from flags_def import FLAGS




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup



otplansampler = OTPlanSampler(method='exact')
def create_training_dataset(ppo_config, batch_size, ppo_resources, gaussians=None):
    """Create training dataset with OT + noise_sampler."""
    datasets = []
    for class_id in tqdm(range(ppo_config.num_classes)):
        targets = ppo_resources['ppo_data']['images'][ppo_resources['class_to_indices'][class_id]].to(device)       
        for target in torch.split(targets, 500):
            if not ppo_config.gaussian:
                target_flip = torch.flip(target, dims=[-1])
                target = torch.cat([target, target_flip], dim=0)
                if ppo_config.std:
                    dist = torch.distributions.Normal(gaussians['means'][class_id].to(device), gaussians['stds'][class_id].to(device))
                    noise = dist.sample((len(target), )).view(-1, 3, 32, 32)
                # gaussians['covs'][class_id] += torch.eye(gaussians['covs'][class_id].size(0)) * 1e-5
                else:
                    dist = torch.distributions.MultivariateNormal(gaussians['means'][class_id].to(device), (gaussians['covs'][class_id] + 1e-4 * torch.eye(3072)).to(device))
                    noise = dist.sample((len(target), )).view(-1, 3, 32, 32)
            else:
                idx = torch.randint(0, len(ppo_resources['ppo_data']['images']), (500,))
                target = ppo_resources['ppo_data']['images'][idx].to(device)
                noise = torch.randn_like(target)
            noise, target = otplansampler.sample_plan(noise, target)
            dataset = torch.utils.data.TensorDataset(noise.cpu(), target.cpu())
            datasets.append(dataset)
    datasets = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    return dataloader


def train(argv):
    if FLAGS.seed!=None:
        torch.manual_seed(FLAGS.seed)
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    run = wandb.init(
        project='AlignFlow-cifar',
        name= FLAGS.wandb_name,
        # name= 'debug',
        config=FLAGS
    )

    # DATASETS/DATALOADER

    if FLAGS.model in ["otcfm", "icfm", "fm", "si"]:
        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            drop_last=True,
        )

        loader_iter = iter(dataloader)
    elif FLAGS.model=='AlignFlow':
        from sdot import SdotPlanSampler
        from sdot_rebalance_dataset import SdotRebalanceDataset
        
        data_tensor=torch.load('./pickled_cifar10_augmented.pt', map_location='cpu').cuda()
        data_tensor_flatten=data_tensor.reshape(data_tensor.shape[0], -1)


        ot_sampler=SdotPlanSampler(data_tensor_flatten, entropic_reg=FLAGS.entropic_reg)
        dual_weight=torch.load('./CIFAR10_dual_weight_flip.pt', map_location='cpu').cuda()
        ot_sampler.set_dual_weight(dual_weight)
        sdot_ds=SdotRebalanceDataset(ot_sampler, FLAGS.resample_interval*len(data_tensor))
        sdot_ds.refresh()

        

        dataloader = torch.utils.data.DataLoader(
            sdot_ds,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            drop_last=True,
            shuffle=True
        )
        loader_iter = iter(dataloader)

    elif FLAGS.model == "cotfm":
        ppo_config = PPOConfig()
        ppo_resources = load_ppo_resources(device=device)
        gaussians = torch.load(ppo_config.gaussians_path)
        dataloader = create_training_dataset(ppo_config, FLAGS.batch_size, ppo_resources, gaussians)
        loader_iter = iter(dataloader)
    else:
        raise NotImplementedError()

    

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model=='AlignFlow':
        FM = ConditionalFlowMatcher(sigma=FLAGS.sigma, space=FLAGS.space)
    elif FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma, space=FLAGS.space)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma, space=FLAGS.space)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma, space=FLAGS.space)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma, space=FLAGS.space)
    elif FLAGS.model == "cotfm":
        FM = ConditionalFlowMatcher(sigma=sigma, space=FLAGS.space)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si', 'cotfm]"
        )

    savedir = os.path.join(FLAGS.output_dir, FLAGS.wandb_name)
    os.makedirs(savedir, exist_ok=True)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=4, #TODO check the result for 4
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.08,
    ).to(
        device
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    start_step = 1
    # Resume training if needed
    if FLAGS.resume_step > 0:
        checkpoint = torch.load(
            os.path.join(
                savedir, f"{FLAGS.model}_cifar10_weights_step_{FLAGS.resume_step}.pt"
            )
        )
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optim.load_state_dict(checkpoint["optim"])
        sched.load_state_dict(checkpoint["sched"])
        start_step = checkpoint["step"] + 1
        print(f"Resumed training from step {FLAGS.resume_step}")

    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))
    cnt = 0
    with trange(start_step, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            try:
                batch = next(loader_iter)
            except StopIteration:
                cnt += 1
                if FLAGS.model=='AlignFlow':
                    dataloader.dataset.refresh()
                if FLAGS.model=='cotfm':
                    dataloader = create_training_dataset(ppo_config, FLAGS.batch_size, ppo_resources, gaussians)
                loader_iter = iter(dataloader) # Reset the iterator
                batch = next(loader_iter)
            if FLAGS.model=='AlignFlow':
                x0, target_idx=batch
                x1 = data_tensor[target_idx].cuda()
                x0 = x0.cuda().reshape(*x1.shape)
            elif FLAGS.model == 'cotfm':
                x0, x1 = batch
                x0 = x0.cuda()
                x1 = x1.cuda()
            else:
                x1, _=batch
                x1=x1.cuda()
                x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            t = t.view(-1, 1, 1, 1)
            if FLAGS.space == 'x':
                vt = (vt - xt) / torch.clip((1 - t), min=0.05) 
            # loss = torch.mean((vt - ut) ** 2)
            
            if FLAGS.weighting == "adaptive":
                per_sample = ((vt - ut) ** 2).view(len(t), -1).sum(dim=-1)
                weights = 1.0 / (per_sample.detach() + 1e-3).pow(FLAGS.adaptive_p)
                loss = (weights * per_sample).mean()
            else:
                loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new
            run.log({"loss": loss.item()}, step=step)

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                if FLAGS.model == 'cotfm':
                    noise = sample_noise(batch_size=64) # [64, 3, 32, 32]
                    net_generated_img=generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal", noise=noise)
                    ema_generated_img=generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema", noise=noise)
                else:
                    net_generated_img=generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                    ema_generated_img=generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                wandb.log({"net_generated_img": wandb.Image(make_grid(net_generated_img))}, step=step)
                wandb.log({"ema_generated_img": wandb.Image(make_grid(ema_generated_img))}, step=step)
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    os.path.join(savedir, f"{FLAGS.model}_cifar10_weights_step_{step}.pt")
                )
                ema_model.eval()
                score=get_fid(ema_model, model=FLAGS.model)
                run.log({"fid": score}, step=step)
                ema_model.train()
                net_model.train()


if __name__ == "__main__":
    app.run(train)