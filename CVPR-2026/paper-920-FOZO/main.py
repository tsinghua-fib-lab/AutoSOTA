import os
import time
import argparse
import random
from importlib import reload, import_module
import copy

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data
from dataset.ImageNetMask import imagenet_r_mask

import torch    
import timm
import numpy as np
from tta_library.FOZO import FOZO
from calibration_library.metrics import ECELoss

from quant_library.quant_utils.models import get_net
from quant_library.quant_utils import net_wrap
import quant_library.quant_utils.datasets as datasets
from quant_library.quant_utils.quant_calib import HessianQuantCalibrator

from models.vpt import PromptViT


def validate_adapt(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')  
    top1 = AverageMeter('Acc@1', ':6.2f')  
    top5 = AverageMeter('Acc@5', ':6.2f')  
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')  
    
    outputs_list, targets_list = [], []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.gpu) 


    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    forward_times = []

    with torch.no_grad(): 
        end = time.time()  
        for i, dl in enumerate(val_loader): 
            images, target = dl[0], dl[1] 
            images = images.cuda()  
            target = target.cuda()  
            if i < 2:
                start_events[i].record()
                output = model(images)
                end_events[i].record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_events[i].elapsed_time(end_events[i])
                forward_times.append(elapsed_time_ms)
                logger.info(f"Forward pass {i+1} time: {elapsed_time_ms:.3f} ms")
            else:
                output = model(images) 

            # -----------------------------------------------------------------
            outputs_list.append(output.cpu()) 
            targets_list.append(target.cpu())  

            acc1, acc5 = accuracy(output, target, topk=(1, 5))  
            top1.update(acc1[0], images.size(0))  
            top5.update(acc5[0], images.size(0))  
            del output  

            batch_time.update(time.time() - end)  
            end = time.time()  
            if i % 5 == 0:  
                logger.info(progress.display(i))  
            
        outputs_list = torch.cat(outputs_list, dim=0).numpy()  
        targets_list = torch.cat(targets_list, dim=0).numpy()  
        
        logits = args.algorithm != 'lame'  
        ece_avg = ECELoss().loss(outputs_list, targets_list, logits=logits) 

    if len(forward_times) >= 2:
        logger.info(f"Summary: 1st forward: {forward_times[0]:.3f} ms, 2nd forward: {forward_times[1]:.3f} ms")

    return top1.avg, top5.avg, ece_avg

def obtain_train_loader(args):
    """Get the training set loader"""
    temp_args = copy.deepcopy(args)
    
    # Ensure obtain_train_loader always loads the complete ImageNet original validation set
    # Temporarily set continual to False and corruption to 'original'
    temp_args.continual = False
    temp_args.corruption = 'original' 
    
    # Call prepare_test_data to get the complete ImageNet validation set
    train_dataset, train_loader = prepare_test_data(temp_args) 
    train_dataset.switch_mode(True, False) 
    return train_dataset, train_loader  

def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk("./quant_library/configs"))  
    if config_name+".py" in files:  
        quant_cfg = import_module(f"quant_library.configs.{config_name}")  
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")  
    reload(quant_cfg)  
    return quant_cfg 

def get_args():
    """Get arguments"""
    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')  # Create argument parser

    # path of data, output dir
    # Data path, output directory
    parser.add_argument('--data', default='/root/autodl-tmp/ILSVRC2012_img_val', help='path to dataset')
    parser.add_argument('--data_v2', default='/root/autodl-tmp/ImageNetV2', help='path to dataset')
    parser.add_argument('--data_sketch', default='/root/autodl-tmp/ImageNet-Sketch', help='path to dataset')
    parser.add_argument('--data_corruption', default='/root/autodl-tmp/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--data_rendition', default='/root/autodl-tmp/imagenet-r', help='path to corruption dataset')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2000, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # algorithm selection
    parser.add_argument('--algorithm', default='fozo', type=str, help='supporting foa, sar, cotta and etc.')
    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    # model settings
    parser.add_argument('--quant', default=False, action='store_true', help='whether to use quantized model in the experiment')
    # output settings
    parser.add_argument('--output', default='./experiment_results', help='the output directory of this experiment')
    parser.add_argument('--tag', default='_first_experiment', type=str, help='the tag of experiment')
    # fozo settings
    parser.add_argument('--num_prompts', default=3, type=int, help='number of inserted prompts for test-time adaptation.')
    parser.add_argument('--fitness_lambda', default=0.4, type=float, help='the balance factor $lambda$')   
    parser.add_argument('--zo_eps', default=0.5, type=float, help='1')
    parser.add_argument('--lr', default=0.08, type=float, help='2')
    parser.add_argument('--n_spsa', default=1, type=int, help='3')
    parser.add_argument('--continual', default=False, action='store_true', help='If true, use robustbench 5k test set for continual evaluation across corruptions.')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    robustbench_data_dir = os.path.join(current_dir, 'robustbench', 'data')
    parser.add_argument('--imagenet_5k_indices_file', 
                        default=os.path.join(robustbench_data_dir, 'imagenet_test_image_ids.txt'), 
                        type=str,
                        help='Path to the file containing 5k ImageNet validation image paths (from RobustBench).')
    parser.add_argument('--imagenet_class_map_file', 
                        default=os.path.join(robustbench_data_dir, 'imagenet_class_to_id_map.json'), 
                        type=str,
                        help='Path to ImageNet class to ID mapping file (from RobustBench).')

    return parser.parse_args() 


if __name__ == '__main__':
    args = get_args()
    
    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    # create logger for experiment
    args.output += '/' + args.algorithm + args.tag + '/'
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    log_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-log.txt"
    logger = get_logger(name="project", output_directory=args.output, log_name=log_name, debug=False)
    original_log_path = os.path.join(args.output, log_name)
    logger.info(args)

    # configure the domains for adaptation
    # options for ImageNet-R/V2/Sketch are ['rendition', 'v2', 'sketch']
    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    # If args.continual is True, the log indicates that a 5k subset of all corruptions will be tested
    if args.continual:
        logger.info("Running in continual mode: Will test all specified corruptions using the 5k ImageNet subset.")
    if args.quant:
        # Use PTQ4Vit for model quantization
        # NOTE the bit of quantization can be modified in quant_library/configs/PTQ4ViT.py
        quant_cfg = init_config("PTQ4ViT")
        net = get_net('vit_base_patch16_224')
        wrapped_modules = net_wrap.wrap_modules_in_net(net,quant_cfg)
        g=datasets.ViTImageNetLoaderGenerator(args.data,'imagenet',32,32,16,kwargs={"model":net})
        test_loader=g.test_loader()
        calib_loader=g.calib_loader(num=32)
        
        quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
        quant_calibrator.batching_quant_calib()
    else:
        # full precision model
        import os as _os
        _os.environ.setdefault('HF_HUB_OFFLINE', '1')
        _os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        try:
            net = timm.create_model('vit_base_patch16_224', pretrained=True)
        except Exception:
            from safetensors.torch import load_file as _load_file
            net = timm.create_model('vit_base_patch16_224', pretrained=False)
            _model_path = '/root/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/snapshots/main/model.safetensors'
            if not _os.path.exists(_model_path):
                _model_path = '/repo/models/vit/model.safetensors'
            _state_dict = _load_file(_model_path)
            net.load_state_dict(_state_dict, strict=True)
        
    net = net.cuda()
    net.eval()
    net.requires_grad_(False)

    
    if args.algorithm == 'fozo':
        # IDEA-013: Compute mean patch embedding for better prompt initialization
        _, train_loader_for_init = obtain_train_loader(args)
        # Use first few batches to estimate mean patch embedding
        patch_embeddings = []
        with torch.no_grad():
            for idx, dl in enumerate(train_loader_for_init):
                if idx >= 10:  # 10 batches = 640 images
                    break
                images = dl[0].cuda()
                patches = net.patch_embed(images)  # (B, N, D)
                patch_embeddings.append(patches.mean(dim=(0, 1)))  # Mean over batch and patches
        if patch_embeddings:
            mean_patch_embed = torch.stack(patch_embeddings).mean(dim=0)  # (D,)
            # Create prompt_init: repeat mean_patch_embed for num_prompts
            prompt_init = mean_patch_embed.unsqueeze(0).unsqueeze(0).repeat(1, args.num_prompts, 1)  # (1, P, D)
            net = PromptViT(net, args.num_prompts, prompt_init=prompt_init).cuda()
        else:
            net = PromptViT(net, args.num_prompts).cuda()
        del train_loader_for_init, patch_embeddings
        adapt_model = FOZO(net, zo_eps=args.zo_eps, lr=args.lr, fitness_lambda=args.fitness_lambda, n_spsa=args.n_spsa)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'no_adapt':
        adapt_model = net
    else:
        assert False, NotImplementedError


    # IDEA-022: Corruption-specific fitness_lambda for optimal entropy/alignment balance
    corruption_lambda = {
        # Noise corruptions: destroy low-level stats → more alignment
        'gaussian_noise': 0.6,
        'shot_noise': 0.6,
        'impulse_noise': 0.6,
        # Blur corruptions: strong feature distortion → more alignment
        'defocus_blur': 0.7,
        'glass_blur': 0.7,
        'motion_blur': 0.7,
        'zoom_blur': 0.7,
        # Weather corruptions: less feature impact → more entropy
        'snow': 0.3,
        'frost': 0.3,
        'fog': 0.3,
        'brightness': 0.3,
        # Digital corruptions: moderate → default
        'contrast': 0.5,
        'elastic_transform': 0.5,
        'pixelate': 0.4,
        'jpeg_compression': 0.4,
    }

    corrupt_acc, corrupt_ece = [], []
    for corrupt in corruptions:
        args.corruption = corrupt
        logger.info(args.corruption)

        if args.corruption == 'rendition':
            adapt_model.imagenet_mask = imagenet_r_mask
        else:
            adapt_model.imagenet_mask = None
        

        # IDEA-022: Set corruption-specific fitness_lambda
        if args.algorithm == 'fozo' and args.corruption in corruption_lambda:
            adapt_model.fitness_lambda = corruption_lambda[args.corruption]
        else:
            adapt_model.fitness_lambda = args.fitness_lambda

        val_dataset, val_loader = prepare_test_data(args)

        torch.cuda.empty_cache()
        top1, top5, ece_loss = validate_adapt(val_loader, adapt_model, args)
        logger.info(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.6f} and Top-5 Accuracy: {top5:.6f} and ECE: {ece_loss:.6f}")
        corrupt_acc.append(top1)
        corrupt_ece.append(ece_loss)

        # reset model before adapting on the next domain
        if args.algorithm == 'no_adapt' or args.continual==True:
            pass
            # adapt_model.reset()
        else:
            adapt_model.reset()
            # pass
        
        logger.info(f'mean acc of corruption: {sum(corrupt_acc)/len(corrupt_acc) if len(corrupt_acc) else 0}')
        logger.info(f'mean ece of corruption: {sum(corrupt_ece)/len(corrupt_ece)*100 if len(corrupt_ece) else 0}')
        logger.info(f'corrupt acc list: {[_.item() for _ in corrupt_acc]}')
        logger.info(f'corrupt ece list: {[_*100 for _ in corrupt_ece]}')

    mean_acc = sum(corrupt_acc)/len(corrupt_acc)
    mean_ece = sum(corrupt_ece)/len(corrupt_ece)*100
    new_log_name = f"{log_name.replace('-log.txt', '')}_acc={mean_acc:.2f}_ece={mean_ece:.2f}.txt"
    new_log_path = os.path.join(args.output, new_log_name)
    os.rename(original_log_path, new_log_path)