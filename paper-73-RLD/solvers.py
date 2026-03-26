import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import AdamW, Adam
from models import GCN, MLP
from torch_scatter import scatter_min, scatter_max, scatter_add
from torch_geometric.utils import add_remaining_self_loops
from accelerate import Accelerator
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import DataLoader
from utils import collate_fn, build_batch
import torch.nn.functional as F
import importlib
import scipy.sparse as sp
from scipy.optimize import minimize

class MetaSolver(object):
    def __init__(self, args):
        self.num_t = args.num_t
        self.num_k = args.num_k
        self.model = None
        self.optimizer = None
        self.skip_decode = args.skip_decode
        self.mixed_precision = args.mixed_precision
        self.model = None
        self.optimizer = None
        
        module = importlib.import_module('.'.join(['problems' ,args.problem]))
        dataset = getattr(module, 'Dataset')
        problem = getattr(module, 'Problem')
        self.problem = problem(args.beta)
        
        if args.mixed_precision == 'no':
            self.dtype = torch.float
        elif args.mixed_precision == 'fp16':
            self.dtype = torch.float16
        elif args.mixed_precision == 'bf16':  
            # Note bf16 does not support kth value
            self.dtype = torch.bfloat16
        else: 
            raise NotImplementedError
        
        if args.do_train:
            self.epochs = args.epochs
            train_dataset = dataset(args.train_path, self.dtype, args.train_samples)
            valid_dataset = dataset(args.valid_path, self.dtype, args.valid_samples)
            self.train_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn) 
            self.valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)
            self.test_loader = None
        else:
            test_dataset = dataset(args.test_path, self.dtype, args.test_samples)
            self.test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)
            self.train_loader = None
            self.valid_loader = None
            
        accelerator = Accelerator(mixed_precision=self.mixed_precision)
        self.accelerator = accelerator
            
    def prepare(self):
        self.model, self.optimzier, self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(self.model, self.optimizer, self.train_loader, self.valid_loader, self.test_loader)
        
    def print_log(self, logger):
        log = []
        for k, v in logger.items():
            log.extend([k + ':', str(v)])
        log = ' '.join(log)
        self.accelerator.print(log)

    def load_ckpt(self, ckpt):
        self.model.load_state_dict(ckpt)

    def save_ckpt(self, save_file):
        torch.save(self.accelerator.get_state_dict(self.model), save_file)
    
    def train_batch(graphs):
        pass

    def evaluate_single(graph):
        pass
    
    def train(self, save_dir):
        best_eval = 0
        num_iter = 0
        logger = {'Epoch': 0, 'Iteration': 0, 'loss': None, 'valid_obj': None, 'valid_time': None}    
        best_eval = np.inf
        iterations = 0
        patience = 10
        wait = 0
        for epoch in range(self.epochs):
            logger['Epoch'] = epoch+1
            self.model.train()
            for index, graphs in tqdm(enumerate(self.train_loader), total=len(self.train_loader), disable=not self.accelerator.is_local_main_process):  
                logger['Iteration'] = index+1
                iterations += 1
                loss = self.train_batch(graphs)
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()   
                if logger['loss'] is None:
                    logger['loss'] = loss.item()
                else:
                    logger['loss'] = 0.95*logger['loss'] + 0.05*loss.item()
                
            self.accelerator.wait_for_everyone() 
            with torch.no_grad():
                obj_result, time_result = self.evaluate('valid')
            logger['valid_obj'] = obj_result
            logger['valid_time'] = time_result
            self.print_log(logger)
            if obj_result < best_eval:
                best_eval = obj_result
                self.save_ckpt(os.path.join(save_dir, 'best.pt'))
                wait = 0
            else:
                wait += 1
            if wait == patience:
                self.accelerator.print('Early stop')
                break

    @torch.no_grad()
    def evaluate(self, mode, log_path=None):
        assert mode in ['valid', 'test']
        if mode == 'valid':
            dataloader = self.valid_loader
        else:
            dataloader = self.test_loader
        if self.model is not None:
            self.model.eval()
        all_ids = []
        all_time = []   
        all_eval_set_size = []
        loss_trackers = []
        for batch_idx, graphs in tqdm(enumerate(dataloader), total=len(dataloader), disable=not self.accelerator.is_local_main_process):
            for graph in graphs: 
                st_time = time.time()
                eval_set_size = self.evaluate_single(graph)
                end_time = time.time()
                all_ids.append(torch.tensor(graph.idx, device=self.accelerator.device))
                all_time.append(torch.tensor(end_time-st_time, device=self.accelerator.device))
                all_eval_set_size.append(eval_set_size)

        all_time = torch.stack(all_time)
        all_time = self.accelerator.gather(all_time)
                               
        all_eval_set_size = torch.stack(all_eval_set_size)        
        all_eval_set_size = self.accelerator.gather(all_eval_set_size)

        obj_result = torch.mean(all_eval_set_size).item()
        time_result = torch.mean(all_time).item()
        
        if log_path is not None:
            if self.accelerator.is_local_main_process:
                with open(log_path, 'w') as f:
                    f.write('Avg. obj: %f\n'%obj_result)
                    f.write('Avg. time: %f\n'%time_result)
                #loss_trackers.dump(log_path.replace('txt', 'npy'))
                self.accelerator.print(obj_result, time_result)
        
        return obj_result, time_result


class RLSA(MetaSolver):
    def __init__(self, args):
        super(RLSA, self).__init__(args)
        self.num_d = args.num_d
        self.tau0 = args.tau0
        
    def evaluate_single(self, graph):
        x = torch.randint(0,2, (graph.num_nodes, self.num_k), device=self.accelerator.device, dtype=graph.edge_weight.dtype)
        A = torch.sparse_coo_tensor(
            graph.edge_index, 
            graph.edge_weight, 
            torch.Size((graph.num_nodes, graph.num_nodes))
        ).to_sparse_csr()
        energy, grad = self.problem.energy_func(A, graph.b, x, True)
        
        best_sol = x.clone()
        best_energy = energy.clone()
        
        restart_interval = 25  # Every 25 steps, refresh worst chains from best
        num_refresh = self.num_k * 3 // 20  # Refresh 15% of chains
        
        for epoch in range(self.num_t):          
            tau = self.tau0*(1-epoch/self.num_t)
            delta = grad*(2*x-1)/2
            
            #The kth value method
            term2 = -torch.kthvalue(-delta, self.num_d, dim=0, keepdim=True).values
            flip_prob = torch.sigmoid((delta-term2)/tau)
            
            rr = torch.rand_like(x)
            x = torch.where(rr<flip_prob, 1-x, x)
            
            energy, grad = self.problem.energy_func(A, graph.b, x, True if epoch < self.num_t-1 else False)
            best_sol = torch.where((energy<best_energy).unsqueeze(0).repeat(graph.num_nodes,1), x, best_sol)
            best_energy = torch.where(energy<best_energy, energy, best_energy)
            
            # Population refresh: replace worst chains from global best solutions
            if (epoch + 1) % restart_interval == 0 and epoch < self.num_t - 1:
                # Sort by GLOBAL best energy (not current x energy)
                sorted_by_best = torch.argsort(best_energy)  # best global per chain
                best_indices = sorted_by_best[:num_refresh]
                worst_indices = sorted_by_best[-num_refresh:]
                
                # Vectorized: get best global solutions and perturb
                x_src = best_sol[:, best_indices]  # use global best, not current
                # Adaptive perturbation: higher early, lower late
                perturb_rate = 0.15 * (1 - epoch / self.num_t) + 0.05
                flip_mask = (torch.rand_like(x_src) < perturb_rate)
                x_refreshed = torch.where(flip_mask, 1 - x_src, x_src)
                x[:, worst_indices] = x_refreshed
                
                # Recompute energy and grad for refreshed chains
                energy, grad = self.problem.energy_func(A, graph.b, x, True)
               
        if self.skip_decode:
            return best_energy.min()
        else:
            # Greedy repair: enforce IS feasibility + add free nodes before decoding
            dtype = best_sol.dtype
            best_sol_repaired = best_sol.clone()
            
            # Check conflicts: nodes adjacent to selected nodes
            conflicts = (A @ best_sol_repaired) * best_sol_repaired  # A already has edge_weight=0.5
            conflicts = (conflicts > 0.4).to(dtype)  # conflict detected
            best_sol_repaired = best_sol_repaired * (1 - conflicts)
            
            # Greedy addition: add free nodes (no neighbor selected)
            for _ in range(3):
                neighbor_count = (A @ best_sol_repaired)
                free_nodes = ((neighbor_count < 0.1) & (best_sol_repaired < 0.5)).to(dtype)
                best_sol_repaired = (best_sol_repaired + free_nodes).clamp(max=1)
            
            best_energy = self.problem.decode_func(best_sol_repaired, graph)
            return best_energy.min()
        

class RLNN(MetaSolver):
    def __init__(self, args):
        super(RLNN, self).__init__(args)
        
        self.model = GCN(1, args.num_h, args.num_l, 1)
        self.model.reset_parameters()
        
        if args.do_train:
            self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        else:
            state_dict = torch.load(
                os.path.join(args.save_dir, 'best.pt'), 
                weights_only=True, 
                map_location=self.accelerator.device
            )
            self.load_ckpt(state_dict)

        self.num_d = args.num_d
        self.lambd = args.lambd
        self.num_tp = args.num_tp
        self.num_kp = args.num_kp
        self.batch_size = args.batch_size
        self.loss = args.loss

    def train(self, save_dir):
        best_eval = 0
        num_iter = 0
        logger = {'Epoch': 0, 'Iteration': 0, 'loss': None, 'valid_obj': None, 'valid_time': None}    
        best_eval = np.inf
        iterations = 0
        self.accelerator.wait_for_everyone() 

        self.print_log(logger)
        for epoch in range(self.epochs):
            logger['Epoch'] = epoch+1
            self.model.train()
            for index, graphs in tqdm(enumerate(self.train_loader), total=len(self.train_loader), disable=not self.accelerator.is_local_main_process): 

                for graph in graphs:
                    inputs = self.generate_data(graph)
                    A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, torch.Size((graph.num_nodes, graph.num_nodes))).to_sparse_csr()
                    num_samples = inputs.shape[1]

                    num_batch = int(np.ceil(num_samples/self.batch_size))
                    for i in range(num_batch):
                        x = inputs[:,i*self.batch_size:(i+1)*self.batch_size]
                        logits = self.model(x.unsqueeze(-1), graph.edge_index).squeeze(-1)
                        prob = torch.sigmoid(logits)
                        if self.loss == 'erdoes':
                            x = (1-prob)*x + prob*(1-x)
                            energy, _ = self.problem.energy_func(A, graph.b, x, False)
                            loss = energy.mean()+self.lambd*((torch.sum(prob,0)-self.num_d)**2).mean()
                        elif self.loss == 'reinforce':
                            m = Bernoulli(prob)
                            change = m.sample()
                            log_prob = m.log_prob(change).sum(0)
                            original_energy, _ = self.problem.energy_func(A, graph.b, x, False)
                            x = (1-change)*x + change*(1-x)
                            energy, _ = self.problem.energy_func(A, graph.b, x, False)
                            reward = original_energy-energy
                            loss = -(log_prob*reward).mean() + self.lambd*((torch.sum(prob,0)-self.num_d)**2).mean()
                        else:
                            raise NotImplementedError
                        
                        self.optimizer.zero_grad()
                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        if logger['loss'] is None:
                            logger['loss'] = loss.item()
                        else:
                            logger['loss'] = 0.95*logger['loss'] + 0.05*loss.item()
                                
                iterations += 1

            self.accelerator.wait_for_everyone() 
            with torch.no_grad():
                obj_result, time_result = self.evaluate('valid')
            logger['valid_obj'] = obj_result
            logger['valid_time'] = time_result
            self.print_log(logger)
            if obj_result < best_eval:
                best_eval = obj_result
                self.save_ckpt(os.path.join(save_dir, 'best.pt'))
    
    def generate_data(self, graph):
        x = torch.randint(0,2, (graph.num_nodes, self.num_k), device=self.accelerator.device, dtype=self.dtype)
        A = torch.sparse_coo_tensor(
            graph.edge_index, 
            graph.edge_weight, 
            torch.Size((graph.num_nodes, graph.num_nodes))
        ).to_sparse_csr()
        data = []

        for epoch in range(self.num_tp):
            data.append(x.clone())
            
            delta = self.model(x.unsqueeze(-1), graph.edge_index).squeeze(-1)
            flip_prob = torch.sigmoid(delta)

            rr = torch.rand_like(x)
            x = torch.where(rr<flip_prob, 1-x, x)
            
        return torch.cat(data, dim=-1)
          
    def evaluate_single(self, graph):
        x = torch.randint(0,2, (graph.num_nodes, self.num_k), device=self.accelerator.device, dtype=self.dtype)
        A = torch.sparse_coo_tensor(
            graph.edge_index, 
            graph.edge_weight, 
            torch.Size((graph.num_nodes, graph.num_nodes))
        ).to_sparse_csr()
        energy, _ = self.problem.energy_func(A, graph.b, x, False)
        
        best_sol = x.clone()
        best_energy = energy.clone()
        
        for epoch in range(self.num_t):          
            delta = self.model(x.unsqueeze(-1), graph.edge_index).squeeze(-1)
            flip_prob = torch.sigmoid(delta)

            rr = torch.rand_like(x)
            x = torch.where(rr<flip_prob, 1-x, x)

            energy, _ = self.problem.energy_func(A, graph.b, x, False)        
            best_sol = torch.where((energy<best_energy).unsqueeze(0).repeat(graph.num_nodes,1), x, best_sol)
            best_energy = torch.where(energy<best_energy, energy, best_energy)

        if self.skip_decode:
            return best_energy.min()
        else:
            # Greedy repair: enforce IS feasibility + add free nodes before decoding
            dtype = best_sol.dtype
            best_sol_repaired = best_sol.clone()
            
            # Check conflicts: nodes adjacent to selected nodes
            conflicts = (A @ best_sol_repaired) * best_sol_repaired  # A already has edge_weight=0.5
            conflicts = (conflicts > 0.4).to(dtype)  # conflict detected
            best_sol_repaired = best_sol_repaired * (1 - conflicts)
            
            # Greedy addition: add free nodes (no neighbor selected)
            for _ in range(3):
                neighbor_count = (A @ best_sol_repaired)
                free_nodes = ((neighbor_count < 0.1) & (best_sol_repaired < 0.5)).to(dtype)
                best_sol_repaired = (best_sol_repaired + free_nodes).clamp(max=1)
            
            best_energy = self.problem.decode_func(best_sol_repaired, graph)
            return best_energy.min()