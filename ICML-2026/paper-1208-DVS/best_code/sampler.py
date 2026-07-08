import os
import time
import pickle
import math
import torch

from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, \
                         load_ema_from_ckpt, load_sampling_fn, load_eval_settings
from utils.graph_utils import adjs_to_graphs, get_init_flags, quantize, quantize_mol, \
                                data_rescale
from utils.plot import save_graph_list, plot_graphs_list
from evaluation.stats import eval_graph_list
from evaluation.molsets import get_all_metrics
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx

# -------- Sampler for generic graph generation tasks --------
class Sampler(object):
    def __init__(self, config):
        super(Sampler, self).__init__()
        self.config = config
        self.device = load_device()

    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']
        load_seed(self.configt.seed)
        self.train_graph_list, self.val_graph_list, self.test_graph_list = load_data(self.configt, get_graph_list=True)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f'{self.config.ckpt}')
            start_log(logger, self.configt)
            train_log(logger, self.configt, self.ckpt_dict['params'])
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model = load_model_from_ckpt(self.ckpt_dict['params'], self.ckpt_dict['state_dict'], self.device)

        if self.config.sample.use_ema:
            self.ema = load_ema_from_ckpt(self.model, self.ckpt_dict['ema'], self.configt.train.ema)
            self.ema.copy_to(self.model.parameters())

        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        NUM_SAMPLES = len(self.test_graph_list)
        num_sampling_rounds = math.ceil(NUM_SAMPLES / self.config.sample.batch_size)
        gen_graph_list = []
        for r in range(num_sampling_rounds):
            self.init_flags = get_init_flags(self.train_graph_list, self.configt, 
                                            self.config.sample.batch_size).to(self.device[0])
            x, adj, (x_conv, adj_conv) = self.sampling_fn(self.model, self.init_flags)
            adj_int = quantize(adj)
            gen_graph_list.extend(adjs_to_graphs(adj_int, True))
        gen_graph_list = gen_graph_list[:NUM_SAMPLES]

        # -------- Evaluation --------
        methods, kernels = load_eval_settings(self.config.data.data, kernel=self.config.sample.kernel)
        result_dict = eval_graph_list(self.test_graph_list, gen_graph_list, methods=methods, kernels=kernels)
        logger.log(f'MMD_full {result_dict}', verbose=False)
        logger.log('='*100)

        # -------- Save samples --------
        save_name = f'{self.log_name}' 
        save_dir = save_graph_list(self.log_folder_name, save_name, gen_graph_list)
        with open(save_dir, 'rb') as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(graphs=sample_graph_list, title=save_name, max_num=16, save_dir=self.log_folder_name)


# -------- Sampler for molecule generation tasks --------
class Sampler_mol(object):
    def __init__(self, config):
        self.config = config
        self.device = load_device()

    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']
        load_seed(self.config.seed)

        train_smiles, test_smiles = load_smiles(self.configt.data.data)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)

        self.train_graph_list, _ = load_data(self.configt, get_graph_list=True)     # for init_flags
        with open(f'data/{self.configt.data.data.lower()}_test_nx.pkl', 'rb') as f:
            self.test_graph_list = pickle.load(f)                                   # for NSPDK MMD

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f'{self.config.ckpt}')
            start_log(logger, self.configt)
            train_log(logger, self.configt, self.ckpt_dict['params'])
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model = load_model_from_ckpt(self.ckpt_dict['params'], self.ckpt_dict['state_dict'], self.device)

        if self.config.sample.use_ema:
            self.ema = load_ema_from_ckpt(self.model, self.ckpt_dict['ema'], self.configt.train.ema)

        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        assert self.config.sample.test_size % self.config.sample.batch_size == 0
        num_sampling_rounds = math.ceil(self.config.sample.test_size / self.config.sample.batch_size)
        gen_smiles = []
        num_mols = 0
        num_mols_valid = 0
        for r in range(num_sampling_rounds):
            self.init_flags = get_init_flags(self.train_graph_list, self.configt, 
                                                self.config.sample.batch_size).to(self.device[0])
            x, adj, _ = self.sampling_fn(self.model, self.init_flags)
            adj = data_rescale(adj, self.configt.model.adj_scale)
            
            samples_int = quantize_mol(adj)
            samples_int = samples_int - 1
            samples_int[samples_int == -1] = 3      # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2
            adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)

            x = torch.where(x > 0.5, 1, 0)
            x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)      # 32, 9, 4 -> 32, 9, 5

            gen_mols, num_mols_wo_correction = gen_mol(x, adj, self.configt.data.data)
            num_mols += len(gen_mols)
            num_mols_valid += num_mols_wo_correction

            smiles = mols_to_smiles(gen_mols)
            smiles = [smi for smi in smiles if len(smi)]
            gen_smiles.extend(smiles)
            
            # -------- Save generated molecules --------
            if not(os.path.isdir(f'./samples/mols/{self.log_folder_name}')):
                os.makedirs(os.path.join(f'./samples/mols/{self.log_folder_name}'))
            with open(os.path.join('samples', 'mols', f'{self.log_folder_name}', f'{self.log_name}.txt'), 'a') as f:
                for smi in smiles:
                    f.write(f'{smi}\n')

        # -------- Evaluation --------
        gen_smiles = gen_smiles[:self.config.sample.test_size]
        scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, 
                                    test=test_smiles, train=train_smiles)
        scores_nspdk = eval_graph_list(self.test_graph_list, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']

        logger.log(f'Num mols: {num_mols}')
        logger.log(f'val w/o corr: {num_mols_valid / num_mols:.4f}')
        for metric in ['FCD/Test', 'Scaf/Test', 'Frag/Test', 'SNN/Test', f'unique@{len(gen_smiles)}', 'Novelty', 'valid']:
            logger.log(f'{metric:12s}: {scores[metric]:.4f}')
        logger.log(f'NSPDK MMD   : {scores_nspdk:.4f}')
        logger.log('='*100)
