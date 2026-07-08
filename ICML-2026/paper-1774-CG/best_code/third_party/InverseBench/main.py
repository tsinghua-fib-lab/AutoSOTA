'''
This file runs algorithms to solve inverse problems and evaluate the results.

Inference steps:
1. instantiate the forward model
2. instantiate the dataloder for test data
3. load the pretrained diffusion model
4. run the inference algorithm

Evaluation steps:
1. instantiate the evaluation metric(s)
2. evaluate the results
'''
import os
from omegaconf import OmegaConf
import pickle
import hydra
from hydra.utils import instantiate


import torch
from torch.utils.data import DataLoader
import wandb

from utils.helper import open_url, create_logger
from tqdm import tqdm

from algo.reinforce import CalibratedGuidanceReinforce
from algo.meanflow_posterior import load_meanflow_net


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config, run_reinforce=True, testloader_portion=0, total_portions=1):
    testloader_portion = int(os.environ.get("TESTLOADER_PORTION", testloader_portion))
    total_portions = int(os.environ.get("TOTAL_PORTIONS", total_portions))
    # DEBUG: Hardcoded parameters
    from omegaconf import open_dict
    with open_dict(config):
        config.problem = OmegaConf.load("configs/problem/blackhole.yaml")
        config.algorithm = OmegaConf.load("configs/algorithm/dps.yaml")
        config.pretrain = OmegaConf.load("configs/pretrain/blackhole.yaml")
    with open_dict(config):
        config.problem.model.root = "data/blackhole/measure"
        config.problem.data.root = os.environ.get("DATA_ROOT", "data/blackhole/valid")
        config.problem.data.id_list = os.environ.get("ID_LIST", "0-4")
        config.algorithm.method.guidance_scale = 0.003
        config.algorithm.method.diffusion_scheduler_config.num_steps = int(os.environ.get("NUM_STEPS", 100))
        config.samples_per_image = int(os.environ.get("SAMPLES_PER_IMAGE", 1))
        config.reinf_k = int(os.environ.get("REINF_K", 32))
        config.inner_loop_steps = 25
        config.use_reparam = os.environ.get("USE_REPARAM", "0") == "1"
        config.use_clamp = os.environ.get("USE_CLAMP", "1") == "1"
        config.save_intermediates = os.environ.get("SAVE_INTERMEDIATES", "0") == "1"
        config.stop_at_sigma = float(os.environ["STOP_AT_SIGMA"]) if "STOP_AT_SIGMA" in os.environ else None
        config.last_hop = os.environ.get("LAST_HOP", "0") == "1"
        config.particle_pool_size = int(os.environ["PARTICLE_POOL"]) if "PARTICLE_POOL" in os.environ else None
        config.pool_no_noise = os.environ.get("POOL_NO_NOISE", "0") == "1"
        config.pool_analytic = os.environ.get("POOL_ANALYTIC", "0") == "1"
        _data_split = os.path.basename(str(config.problem.data.root).rstrip("/"))
        _estimator_tag = "reparam" if config.use_reparam else "reinforce"
        _clamp_tag = "" if config.use_clamp else "_noclamp"
        _hop_tag = ""
        if config.stop_at_sigma is not None:
            _hop_tag += f"_stopat{config.stop_at_sigma}"
        if config.last_hop:
            _hop_tag += "_lasthop"
        if config.particle_pool_size is not None:
            _hop_tag += f"_pool{config.particle_pool_size}"
            if config.pool_no_noise:
                _hop_tag += "_nonoise"
        if config.pool_analytic:
            _hop_tag += "_poolinf"
        config.exp_name = f"steps_{config.algorithm.method.diffusion_scheduler_config.num_steps}_reinfk_{config.reinf_k}_meanflow_{_estimator_tag}{_clamp_tag}{_hop_tag}_guidancescale_0.003_{_data_split}"
        config.use_meanflow = True
        config.mf_ckpt = os.environ.get("MF_CKPT", "checkpoints/mean-flow.pt")
        config.mf_easy_meanflow_root = os.environ.get("MF_EASY_MEANFLOW_ROOT", "")
        config.mf_sibling_renoise_t = None  # None -> renoise to current tau
    # END DEBUG

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.tf32:
        torch.set_float32_matmul_precision("high")
    # set random seed
    torch.manual_seed(config.seed)

    if config.wandb:
        problem_name = config.get('problem')['name']
        wandb.init(project=problem_name, group=config.algorithm.name, 
                   config=OmegaConf.to_container(config), 
                   reinit=True, settings=wandb.Settings(start_method="fork"))
        config = OmegaConf.create(dict(wandb.config)) # necessary for wandb sweep because wandb.config will be overwritten by sweep agent right after wandb.init
    # set up directory for logging and saving data
    exp_dir = os.path.join(config.problem.exp_dir, config.algorithm.name, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    # save config
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))
    forward_op = instantiate(config.problem.model, device=device)
    testset = instantiate(config.problem.data)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    logger.info(f"Loaded {len(testset)} test samples...")
    # load pre-trained model
    ckpt_path = config.problem.prior

    try:
        with open_url(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            net = ckpt['ema'].to(device)
    except:
        net = instantiate(config.pretrain.model)
        ckpt = torch.load(config.problem.prior, map_location=device)
        # net.model.load_state_dict(ckpt)
        if 'ema' in ckpt.keys():
            net.load_state_dict(ckpt['ema'])
        else:
            net.load_state_dict(ckpt['net'])
        net = net.to(device)

    del ckpt
    net.eval()
    if config.compile:
        net = torch.compile(net)
    logger.info(f"Loaded pre-trained model from {config.problem.prior}...")
    # set up algorithm
    if not run_reinforce:
        algo = instantiate(config.algorithm.method, forward_op=forward_op, net=net)
    else:
        mf_net = None
        if config.get("use_meanflow", False):
            mf_net = load_meanflow_net(
                config.mf_ckpt, device,
                easy_meanflow_root=config.get("mf_easy_meanflow_root", None),
            )
            logger.info(f"Loaded MeanFlow net from {config.mf_ckpt}")
        algo = CalibratedGuidanceReinforce(
            net, forward_op,
            config.algorithm.method.diffusion_scheduler_config,
            reinf_k=config.reinf_k,
            desired_number_of_steps_inner_loop=config.inner_loop_steps,
            mf_net=mf_net,
            mf_sibling_renoise_t=config.get("mf_sibling_renoise_t", None),
            reparam=config.get("use_reparam", False),
            use_clamp=config.get("use_clamp", True),
            save_intermediates=config.get("save_intermediates", False),
            stop_at_sigma=config.get("stop_at_sigma", None),
            last_hop=config.get("last_hop", False),
            particle_pool_size=config.get("particle_pool_size", None),
            pool_no_noise=config.get("pool_no_noise", False),
            pool_analytic=config.get("pool_analytic", False),
        )
    # set up evaluator

    evaluator = instantiate(config.problem.evaluator, forward_op=forward_op)

    portion_size = len(testset) // total_portions
    start_idx = testloader_portion * portion_size
    end_idx = (testloader_portion + 1) * portion_size if testloader_portion < total_portions - 1 else len(testset)
    print(f"Evaluating portion {testloader_portion+1}/{total_portions}, from index {start_idx} to {end_idx}...")

    for i, data in tqdm(enumerate(testloader)):
        if not start_idx <= i < end_idx:
            print(f"Skipping test sample {i}...")
            continue
        print(f"Processing test sample {i}...")

        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, dict):
            assert 'target' in data.keys(), "'target' must be in the data dict"
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
        data_id = testset.id_list[i]
        for sample_idx in tqdm(range(config.samples_per_image)):
            logger.info(f'Processing test sample {data_id}, sample {sample_idx+1}/{config.samples_per_image}...')
            save_path = os.path.join(exp_dir, f'result_{data_id}_sample_{sample_idx}.pt')
            if config.inference:
                # get the observation
                observation = forward_op(data)
                target = data['target']
                # run the algorithm
                logger.info(f'Running inference on test sample {data_id}...')
                recon = algo.inference(observation, num_samples=config.num_samples, gen_idx=i, guidance_scale=config.algorithm.method.guidance_scale, sample_idx=sample_idx)
                logger.info(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')

                result_dict = {
                    'observation': observation,
                    'generated_sample': recon,
                    'recon': forward_op.unnormalize(recon).cpu(),
                    'target': forward_op.unnormalize(target).cpu(),
                }
                torch.save(result_dict, save_path)
                logger.info(f"Saved results to {save_path}.")
            else:
                # load the results
                result_dict = torch.load(save_path)
                logger.info(f"Loaded results from {save_path}.")

        # evaluate the results
        metric_dict = evaluator(pred=result_dict['recon'], target=result_dict['target'], observation=result_dict['observation'])
        logger.info(f"Metric results: {metric_dict}...")

    logger.info("Evaluation completed...")
    # aggregate the results
    metric_state = evaluator.compute()
    logger.info(f"Final metric results: {metric_state}...")
    if config.wandb:
        wandb.log(metric_state)
        wandb.finish()


if __name__ == "__main__":
    main()
