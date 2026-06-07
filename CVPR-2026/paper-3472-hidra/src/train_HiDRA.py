import os

from torch.autograd import Variable

from DEM import DEM
from StableDiffusionDAloRA import StableDiffusionDAloRA


os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

import gc
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from my_utils.training_utils import parse_args_paired_training, InfraredPairedDataset



def get_lower_loss_params(lower_loss, lower_params):
    vector = []
    for v in lower_params:
        if v.grad is not None:
            vector.append(v.grad.data)
        else:
            vector.append(torch.zeros_like(v))
    dfy = torch.autograd.grad(lower_loss, lower_params, allow_unused=True)
    gfyfy = 0
    gFyfy = 0
    for f, F in zip(dfy, vector):
        if f is None:
            f = torch.zeros_like(F)
        gfyfy = gfyfy + torch.sum(f * f)
        gFyfy = gFyfy + torch.sum(F * f)
    return gfyfy, gFyfy


def backward_step_unrolled(unrolled_loss, lower_loss, upper_params, gFyfy, gfyfy, eta):
    unrolled_loss.backward()
    dalpha = [v.grad for v in upper_params]
    # print(dalpha)
    GN_loss = -gFyfy.detach() / gfyfy.detach() * lower_loss
    implicit_grads1 = torch.autograd.grad(GN_loss, upper_params, allow_unused=True)
    for g, ig in zip(dalpha, implicit_grads1):
        if ig is None:
            ig = torch.zeros_like(g)
        g.data.sub_(ig.data, alpha=eta)

    for v, g2 in zip(upper_params, dalpha):
        if v.grad is None:
            v.grad = Variable(g2.data)
        else:
            v.grad.data.copy_(g2.data)


def main(args):
    # init and save configs
    # config = OmegaConf.load(args.base_config)

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # initialize degradation estimation network
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').cuda().eval()
    net_hyper = DEM(backbone=backbone)
    net_hyper.backbone.requires_grad_(False)
    if args.pretrain_dem_path is not None:
        net_hyper.load_model(args.pretrain_dem_path)
    else:
        print("Initializing hyper prompt model with random weights")
    net_hyper = net_hyper.cuda()
    net_hyper.train()

    # initialize net
    net_enh = StableDiffusionDAloRA(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=sd_path,
                    pretrained_path=args.pretrained_path)
    net_enh.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_enh.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_enh.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='dino', output_type='conv_multi_level',
                                                   loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    layers_to_opt = layers_to_opt + list(net_enh.vae_block_embeddings.parameters()) + list(
        net_enh.unet_block_embeddings.parameters())
    layers_to_opt = layers_to_opt + list(net_enh.vae_de_mlp.parameters()) + list(net_enh.unet_de_mlp.parameters()) + \
                    list(net_enh.vae_block_mlp.parameters()) + list(net_enh.unet_block_mlp.parameters()) + \
                    list(net_enh.vae_fuse_mlp.parameters()) + list(net_enh.unet_fuse_mlp.parameters())

    for n, _p in net_enh.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_enh.unet.conv_in.parameters())

    for n, _p in net_enh.vae.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    dataset_train = InfraredPairedDataset(args.dataset_folder, "train", args.train_image_prep, net_enh.tokenizer)
    dataset_train.set_deg()
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.dataloader_num_workers)
    dataset_val = InfraredPairedDataset(args.dataset_folder, "val", args.test_image_prep, net_enh.tokenizer)
    dataset_val.set_deg()
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
                                       betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                       eps=args.adam_epsilon, )
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)

    optimizer_hyper = torch.optim.AdamW(net_hyper.adapter.parameters(), lr=args.learning_rate,
                                       betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                       eps=args.adam_epsilon, )
    lr_scheduler_hyper = get_scheduler(args.lr_scheduler, optimizer=optimizer_hyper,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)

    # Prepare everything with our `accelerator`.
    net_enh, net_disc, optimizer, optimizer_disc, optimizer_hyper, dl_train, lr_scheduler, lr_scheduler_disc, lr_scheduler_hyper = accelerator.prepare(
        net_enh, net_disc, optimizer, optimizer_disc, optimizer_hyper, dl_train, lr_scheduler, lr_scheduler_disc, lr_scheduler_hyper
    )
    net_hyper, net_lpips = accelerator.prepare(net_hyper, net_lpips)
    # # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_enh.to(accelerator.device, dtype=weight_dtype)
    net_hyper.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, )

    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # start the training loop
    global_step = 0#21001## 0
    upper_updating_step = 4
    joint_steps_bound = 5000 # 5000
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_hyper, net_enh, net_disc]
            with accelerator.accumulate(*l_acc):
                
                if len(batch[0].shape) == 5:
                    # print(global_step+1)
                    # optimizer_hyper.param_groups[0]['lr'] = args.learning_rate / 10
                    # print(optimizer_hyper.param_groups[0]['lr'])
                    
                    x_level_src, x_src, x_tgt, txt_prompt, txt_level_prompt = batch
                    # x_level_src = torch.cat([x_tgt.reshape(B, 1, C, H, W), x_level_src], dim=1)
                    B, L, C, H, W = x_level_src.shape
                    x_level_tgt = torch.stack([x_tgt] * L, dim=1)
                    x_level_src = x_level_src.reshape(B * L, C, H, W)
                    x_level_tgt = x_level_tgt.reshape(B * L, C, H, W)
                    txt_level_prompt = list(np.array(txt_level_prompt).flat)
                    x_level_src, x_level_tgt = x_level_src.cuda(), x_level_tgt.cuda()
                    # B = B * L

                    level_prompt = net_hyper(x_level_src.detach())

                    x_level_tgt_pred = net_enh(x_level_src.detach(), level_prompt, txt_level_prompt)
                    loss_level_l2 = F.mse_loss(x_level_tgt_pred.float(), x_level_tgt.detach().float(), reduction="mean") * args.lambda_l2
                    loss_level_lpips = net_lpips(x_level_tgt_pred.float(), x_level_tgt.detach().float()).mean() * args.lambda_lpips

                    # loss_hyper = loss_dt(net_dt(x_tgt_pred), label)
                    loss_level = loss_level_l2 + loss_level_lpips
                    
                    x_src, x_tgt = x_src.cuda(), x_tgt.cuda()
                    prompt = net_hyper(x_src.detach())
                    x_tgt_pred = net_enh(x_src.detach(), prompt, txt_prompt)
                    loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean") * args.lambda_l2
                    loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean() * args.lambda_lpips

                    loss = loss_l2 + loss_lpips

                    gfyfy, gFyfy = get_lower_loss_params(loss, layers_to_opt)

                    prompt = net_hyper(x_src.detach())
                    x_tgt_pred = net_enh(x_src.detach(), prompt, txt_prompt)
                    loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean") * args.lambda_l2
                    loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean() * args.lambda_lpips
                    loss = loss_l2 + loss_lpips
                    backward_step_unrolled(loss_level, loss, list(p for p in net_hyper.adapter.parameters() if p.requires_grad), gFyfy, gfyfy, optimizer_hyper.param_groups[0]['lr']/5)

                    # accelerator.backward(loss_level, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_hyper.adapter.parameters(), args.max_grad_norm)
                    optimizer_hyper.step()
                    lr_scheduler_hyper.step()
                    optimizer_hyper.zero_grad(set_to_none=args.set_grads_to_none)
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                    
                    
                    """
                    Generator loss: fool the discriminator
                    """
                    # optimizer_hyper.param_groups[0]['lr'] = args.learning_rate / 10
                    level_prompt = net_hyper(x_level_src.detach())
                    x_level_tgt_pred = net_enh(x_level_src.detach(), level_prompt, txt_level_prompt)
                    lossG_level = net_disc(x_level_tgt_pred, for_G=True).mean() * args.lambda_gan
                    
                    prompt = net_hyper(x_src.detach())
                    x_tgt_pred = net_enh(x_src.detach(), prompt, txt_prompt)
                    lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan

                    gfyfy, gFyfy = get_lower_loss_params(lossG, layers_to_opt)
                    prompt = net_hyper(x_src.detach())
                    x_tgt_pred = net_enh(x_src.detach(), prompt, txt_prompt)
                    lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan

                    backward_step_unrolled(lossG_level, lossG, list(p for p in net_hyper.adapter.parameters() if p.requires_grad), gFyfy, gfyfy,
                                           optimizer_hyper.param_groups[0]['lr']/5)
                    
                    # accelerator.backward(lossG_level)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_hyper.adapter.parameters(), args.max_grad_norm)
                    optimizer_hyper.step()
                    lr_scheduler_hyper.step()
                    optimizer_hyper.zero_grad(set_to_none=args.set_grads_to_none)
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                    
                    ########################################################
                    # x_src, x_tgt = x_src.cuda(), x_tgt.cuda()
                    B, C, H, W = x_src.shape
                    prompt = net_hyper(x_src.detach())
                    x_tgt_pred = net_enh(x_src.detach(), prompt, txt_prompt)
                    loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean") * args.lambda_l2
                    loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean() * args.lambda_lpips

                    loss = loss_l2 + loss_lpips

                    accelerator.backward(loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer_hyper.zero_grad(set_to_none=args.set_grads_to_none)
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    """
                    Generator loss: fool the discriminator
                    """
                    prompt = net_hyper(x_src.detach())
                    x_tgt_pred = net_enh(x_src.detach(), prompt, txt_prompt)
                    lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                    accelerator.backward(lossG)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer_hyper.zero_grad(set_to_none=args.set_grads_to_none)
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    """
                    Discriminator loss: fake image vs real image
                    """
                    # real image
                    lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                    accelerator.backward(lossD_real.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    # fake image
                    lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                    accelerator.backward(lossD_fake.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD = lossD_real + lossD_fake
                    
                    

                else:
                    x_level_src = None
                    x_src, x_tgt, txt_prompt = batch
                    x_src, x_tgt = x_src.cuda(), x_tgt.cuda()
                    B, C, H, W = x_src.shape
                    prompt = net_hyper(x_src.detach())
                    x_tgt_pred = net_enh(x_src.detach(), prompt, txt_prompt)
                    loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean") * args.lambda_l2
                    loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean() * args.lambda_lpips

                    loss = loss_l2 + loss_lpips

                    accelerator.backward(loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                        accelerator.clip_grad_norm_(net_hyper.adapter.parameters(), args.max_grad_norm)
                    if  global_step < joint_steps_bound:
                        optimizer_hyper.step() 
                        lr_scheduler_hyper.step()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer_hyper.zero_grad(set_to_none=args.set_grads_to_none)
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    """
                    Generator loss: fool the discriminator
                    """
                    prompt = net_hyper(x_src.detach())
                    x_tgt_pred = net_enh(x_src.detach(), prompt, txt_prompt)
                    lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                    accelerator.backward(lossG)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                        accelerator.clip_grad_norm_(net_hyper.adapter.parameters(), args.max_grad_norm)
                    if  global_step < joint_steps_bound:
                        optimizer_hyper.step() 
                        lr_scheduler_hyper.step()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer_hyper.zero_grad(set_to_none=args.set_grads_to_none)
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    """
                    Discriminator loss: fake image vs real image
                    """
                    # real image
                    lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                    accelerator.backward(lossD_real.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    # fake image
                    lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                    accelerator.backward(lossD_fake.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD = lossD_real + lossD_fake
                    
                    loss_level = None




            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                dataset_train.set_deg()
                if global_step > joint_steps_bound and (global_step + 1) % upper_updating_step == 0:
                    dataset_train.set_level_deg_params()
                    dataset_train.set_level_deg()
                

                if accelerator.is_main_process:
                    logs = {}
                    if loss_level is not None:
                        logs["lossG_level"] = lossG_level.detach().item()
                        logs["loss_level_l2"] = loss_level_l2.detach().item()
                        logs["loss_level_lpips"] = loss_level_lpips.detach().item()
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    progress_bar.set_postfix(**logs)


                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        if x_level_src is not None:
                            log_dict["train/level_source"] = [wandb.Image(x_level_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B * L)]
                            log_dict["train/level_model_output"] = [wandb.Image(x_level_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B * L)]
                        
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        adapter_outf = os.path.join(args.output_dir, "checkpoints", f"model_adapter_{global_step}.pkl")
                        accelerator.unwrap_model(net_enh).save_model(outf)
                        
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_hyper).save_model(adapter_outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips = [], []

                        val_count = 0
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src, x_tgt, txt_prompt = batch_val
                            x_src, x_tgt = x_src.cuda(), x_tgt.cuda()
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                prompt = net_hyper(x_src.detach())

                                x_tgt_pred = accelerator.unwrap_model(net_enh)(x_src.detach(), prompt, txt_prompt)
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())

                            if args.save_val and val_count < 5:
                                x_src = x_src.cpu().detach() * 0.5 + 0.5
                                x_tgt = x_tgt.cpu().detach() * 0.5 + 0.5
                                x_tgt_pred = x_tgt_pred.cpu().detach() * 0.5 + 0.5

                                combined = torch.cat([x_src, x_tgt_pred, x_tgt], dim=3)
                                output_pil = transforms.ToPILImage()(combined[0])
                                outf = os.path.join(args.output_dir, f"val_{step}.png")
                                output_pil.save(outf)
                                val_count += 1

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
