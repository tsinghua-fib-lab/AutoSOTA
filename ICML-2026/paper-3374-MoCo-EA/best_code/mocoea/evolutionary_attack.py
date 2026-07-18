"""Evolutionary attack with traditional and Bezier crossover."""

import torch
import torch.nn as nn
import numpy as np
import time
import math
from typing import List, Tuple, Dict

from mocoea.bezier import BezierAdversarialUnconstrained

class EvolutionaryAttack:
    def __init__(self, model, eps=8/255, norm='linf',
                 population_size=50, elite_size=10,
                 mutation_rate=0.1, mutation_strength=0.02,
                 normalize_fn=None, **kwargs):
        self.model = model
        self.eps = eps
        self.norm = norm
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength * eps
        self.normalize = normalize_fn or (lambda x: x)
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_count = 0
        
        # Feature flags for optimization ideas
        self.flags = kwargs.get('feature_flags', {})
        self._flag = lambda name, default=True: self.flags.get(name, default)
        
        # Per-phase query breakdown (CODE-02)
        self.query_breakdown = {
            'pop_init': 0, 'fitness_eval': 0, 'bezier_opt': 0,
            'bezier_candidate_eval': 0, 'mutation_eval': 0
        }
        
        # Bezier warm-start cache (CODE-03)
        self.cached_theta_ema = None
        self.theta_alpha = 0.0
        
        # Adaptive mutation tracking (ALGO-06)
        self.attack_count = 0
        
        self.bezier = BezierAdversarialUnconstrained(
            model, norm=norm, eps=eps, lr=0.1, num_iter=5,
            normalize_fn=self.normalize
        )
    
    def initialize_population(self, x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        population = []
        initial_strength = 0.2
        
        # ALGO-01: PGD warm-start population initialization
        if self._flag('warm_start', True):
            from mocoea.attacks import PGDAttack
            pgd_iters = self.flags.get('pgd_init_iters', 10) if self.flags else 10
            pgd_alpha = self.eps / 8
            
            for i in range(self.population_size):
                pgd = PGDAttack(self.model, eps=self.eps, alpha=pgd_alpha, 
                               num_iter=pgd_iters, norm=self.norm,
                               randomize=True, normalize_fn=self.normalize)
                with torch.enable_grad():
                    delta = pgd.perturb(x, y)
                population.append(delta.detach())
                self.query_count += pgd_iters  # Count PGD forward passes
                self.query_breakdown['pop_init'] += pgd_iters
        else:
            while len(population) < self.population_size:
                if self.norm == 'linf':
                    delta = torch.empty_like(x, device=self.device).uniform_(-self.eps * initial_strength,
                                                         self.eps * initial_strength)
                elif self.norm == 'l2':
                    delta = torch.randn_like(x, device=self.device)
                    delta = delta / (torch.norm(delta.flatten()) + 1e-10) * self.eps * initial_strength
                else: 
                    delta = torch.randn_like(x, device=self.device) * self.eps * initial_strength * 0.1
                    delta = self.bezier.project_norm_ball(delta * 5)
                
                population.append(delta)
        
        return population
    
    def evaluate_fitness(self, population: List[torch.Tensor], 
                        x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        fitness_scores = []
        
        # CODE-01: Progressive two-stage candidate evaluation
        if self._flag('progressive_eval', True) and len(population) > 10:
            survival_ratio = 0.6
            all_outputs = []
            proxy_losses = []
            for delta in population:
                x_adv = torch.clamp(x + delta, 0, 1)
                with torch.no_grad():
                    outputs = self.model(self.normalize(x_adv))
                    ce_loss = nn.CrossEntropyLoss()(outputs, y).item()
                all_outputs.append(outputs)
                proxy_losses.append(ce_loss)
                self.query_count += 1
                self.query_breakdown['fitness_eval'] += 1
            
            n_survive = max(self.elite_size * 2, int(survival_ratio * len(population)))
            top_indices = set(np.argsort(proxy_losses)[-n_survive:].tolist())
            
            for i in range(len(population)):
                if i in top_indices:
                    outputs = all_outputs[i]
                    pred = outputs.argmax(dim=1)
                    probs = torch.softmax(outputs, dim=1)
                    correct_prob = probs[0, y].item()
                    if pred != y:
                        fitness = 2.0 + (1.0 - correct_prob)
                    else:
                        fitness = 1.0 - correct_prob
                    fitness_scores.append(fitness)
                else:
                    fitness_scores.append(0.0)
        else:
            for delta in population:
                x_adv = torch.clamp(x + delta, 0, 1)
                with torch.no_grad():
                    outputs = self.model(self.normalize(x_adv))
                    pred = outputs.argmax(dim=1)
                    
                    probs = torch.softmax(outputs, dim=1)
                    correct_prob = probs[0, y].item()
                    
                    if pred != y:
                        fitness = 2.0 + (1.0 - correct_prob)
                    else:
                        fitness = 1.0 - correct_prob
                
                fitness_scores.append(fitness)
                self.query_count += 1
                self.query_breakdown['fitness_eval'] += 1
        
        return np.array(fitness_scores)
    
    def traditional_crossover(self, parent1: torch.Tensor, 
                            parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.rand_like(parent1, device=self.device) > 0.5
        child1 = torch.where(mask, parent1, parent2)
        child2 = torch.where(mask, parent2, parent1)
        
        child1 = self.bezier.project_norm_ball(child1)
        child2 = self.bezier.project_norm_ball(child2)
        
        return child1, child2
    
    def bezier_crossover(self, parent1: torch.Tensor, parent2: torch.Tensor,
                        x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # CODE-03: Bezier control-point warm-start cache
        if self._flag('bezier_warm_start', True) and self.cached_theta_ema is not None:
            theta_init = ((1 - self.theta_alpha) * ((parent1 + parent2) / 2) + 
                         self.theta_alpha * self.cached_theta_ema)
        else:
            theta_init = (parent1 + parent2) / 2
        theta = theta_init.clone().detach().requires_grad_(True)
        
        # ALGO-05: Higher momentum in Adam (beta1=0.99 vs 0.9)
        use_momentum = self._flag('bezier_momentum', True)
        lr = 0.2
        beta1 = 0.99 if use_momentum else 0.9
        optimizer = torch.optim.Adam([theta], lr=lr, betas=(beta1, 0.999))
        
        # ALGO-02: Adaptive step count with cosine LR and early stopping
        use_early_stop = self._flag('bezier_early_stop', True)
        max_steps = 5
        min_steps = 2
        patience = self.flags.get('bezier_early_stop_patience', 2) if self.flags else 2
        
        # ALGO-04: Reduced t-sampling after first step
        use_sparse_t = not self._flag('bezier_full_t_samples', False)
        
        prev_loss = float('inf')
        plateau_count = 0
        
        for step in range(max_steps):
            optimizer.zero_grad()
            loss_total = 0
            
            if use_sparse_t and step > 0:
                t_values = torch.tensor([0.25, 0.75]).to(self.device)
            else:
                t_values = torch.tensor([0.25, 0.5, 0.75]).to(self.device)
            
            for t in t_values:
                delta_t = self.bezier.bezier_curve(parent1, theta, parent2, t.item())
                delta_t = self.bezier.project_norm_ball(delta_t)
                x_adv = torch.clamp(x + delta_t, 0, 1)
                outputs = self.model(self.normalize(x_adv))
                
                loss = -nn.CrossEntropyLoss()(outputs, y)
                loss_total += loss
                self.query_count += 1
                self.query_breakdown['bezier_opt'] += 1
            
            # ALGO-02: Cosine LR schedule
            if use_early_stop:
                cos_lr = lr * 0.5 * (1 + math.cos(math.pi * step / max_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cos_lr
            
            loss_total.backward()
            optimizer.step()
            
            # ALGO-02: Early stopping on loss plateau
            if use_early_stop:
                curr_loss = loss_total.item()
                if abs(curr_loss - prev_loss) < 1e-3:
                    plateau_count += 1
                else:
                    plateau_count = 0
                prev_loss = curr_loss
                if step >= min_steps - 1 and plateau_count >= patience:
                    break
        
        theta = theta.detach()
        
        # CODE-03: Update EMA cache
        if self._flag('bezier_warm_start', True):
            if self.cached_theta_ema is None:
                self.cached_theta_ema = theta.clone()
            else:
                self.cached_theta_ema = 0.95 * self.cached_theta_ema + 0.05 * theta
        
        candidates_left = []
        candidates_right = []
        
        for t in [0.1, 0.25, 0.4]:
            delta_t = self.bezier.bezier_curve(parent1, theta, parent2, t)
            delta_t = self.bezier.project_norm_ball(delta_t)
            fitness = self._evaluate_single_fitness(x + delta_t, y)
            candidates_left.append((delta_t, fitness))
            self.query_count += 1
            self.query_breakdown['bezier_candidate_eval'] += 1
        
        for t in [0.6, 0.75, 0.9]:
            delta_t = self.bezier.bezier_curve(parent1, theta, parent2, t)
            delta_t = self.bezier.project_norm_ball(delta_t)
            fitness = self._evaluate_single_fitness(x + delta_t, y)
            candidates_right.append((delta_t, fitness))
            self.query_count += 1
            self.query_breakdown['bezier_candidate_eval'] += 1
        
        child1 = max(candidates_left, key=lambda x: x[1])[0]
        child2 = max(candidates_right, key=lambda x: x[1])[0]
        
        return child1, child2
    
    def _evaluate_single_fitness(self, x_adv: torch.Tensor, y: torch.Tensor) -> float:
        x_adv = torch.clamp(x_adv, 0, 1)
        with torch.no_grad():
            outputs = self.model(self.normalize(x_adv))
            pred = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)
            correct_prob = probs[0, y].item()
            
            if pred != y:
                fitness = 2.0 + (1.0 - correct_prob)
            else:
                fitness = 1.0 - correct_prob
        
        return fitness
    
    def mutate(self, individual: torch.Tensor, x: torch.Tensor = None, y: torch.Tensor = None) -> torch.Tensor:
        # ALGO-06: Use adaptive mutation rate if not fixed
        if self._flag('fixed_mutation', False):
            current_rate = self.mutation_rate
        else:
            current_rate = getattr(self, '_adaptive_mutation_rate', self.mutation_rate)
            
        if torch.rand(1).item() < current_rate:
            # ALGO-03: Saliency-guided mutation
            if self._flag('saliency_mutate', False) and x is not None and y is not None:
                x_adv = torch.clamp(x + individual, 0, 1).clone().detach().requires_grad_(True)
                outputs = self.model(self.normalize(x_adv))
                loss = nn.CrossEntropyLoss()(outputs, y)
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                saliency = torch.abs(grad)
                saliency = saliency / (saliency.max() + 1e-8)
                self.query_count += 1
                self.query_breakdown['mutation_eval'] += 1
                noise_saliency = saliency * torch.randn_like(individual) * self.mutation_strength
                noise_uniform = torch.randn_like(individual) * self.mutation_strength * 0.3
                noise = 0.7 * noise_saliency + 0.3 * noise_uniform
            else:
                noise = torch.randn_like(individual) * self.mutation_strength
            individual = individual + noise
            individual = self.bezier.project_norm_ball(individual)
        return individual
    
    def selection(self, population: List[torch.Tensor], 
                 fitness: np.ndarray) -> List[torch.Tensor]:
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = fitness[tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].clone())
        
        return selected
    
    def evolve(self, x: torch.Tensor, y: torch.Tensor, 
              max_generations: int = 100,
              crossover_type: str = 'traditional',
              early_stop_fitness: float = 2.0) -> Dict:
        start_time = time.time()
        population = self.initialize_population(x, y)
        
        stats = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'success': [],
            'query_counts': [],
            'time_elapsed': []
        }
        
        best_perturbation = None
        best_fitness_ever = -float('inf')
        
        for gen in range(max_generations):
            fitness = self.evaluate_fitness(population, x, y)
            
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            avg_fitness = np.mean(fitness)
            
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
                best_perturbation = population[best_idx].clone()
            
            success = best_fitness >= early_stop_fitness
            
            stats['generations'].append(gen)
            stats['best_fitness'].append(float(best_fitness))
            stats['avg_fitness'].append(float(avg_fitness))
            stats['success'].append(success)
            stats['query_counts'].append(self.query_count)
            stats['time_elapsed'].append(time.time() - start_time)
            
            if success:
                print(f"  Attack successful at generation {gen} (fitness={best_fitness:.3f})")
                break
            
            # ALGO-06: Adaptive mutation rate based on fitness diversity
            if not self._flag('fixed_mutation', False):
                fitness_std = np.std(fitness)
                fitness_mean = np.mean(fitness) + 1e-8
                cv = fitness_std / fitness_mean
                self._adaptive_mutation_rate = max(0.05, min(0.5, 0.2 * (1 + cv)))
            
            # CODE-03: Gradually increase theta warm-start alpha
            if self._flag('bezier_warm_start', True):
                self.theta_alpha = min(0.3, self.attack_count / 10.0 * 0.3)
            
            parents = self.selection(population, fitness)
            
            offspring = []
            for i in range(0, len(parents)-1, 2):
                if crossover_type == 'traditional':
                    child1, child2 = self.traditional_crossover(parents[i], parents[i+1])
                elif crossover_type == 'bezier':
                    child1, child2 = self.bezier_crossover(parents[i], parents[i+1], x, y)
                else:
                    raise ValueError(f"Unknown crossover type: {crossover_type}. Use 'traditional' or 'bezier'")
                
                child1 = self.mutate(child1, x, y)
                child2 = self.mutate(child2, x, y)
                offspring.extend([child1, child2])
            
            elite_idx = np.argsort(fitness)[-self.elite_size:]
            elite = [population[i].clone() for i in elite_idx]
            
            population = elite + offspring[:self.population_size - self.elite_size]
        
        stats['best_perturbation'] = best_perturbation
        stats['final_generation'] = gen
        
        # Increment attack count for adaptive features
        self.attack_count += 1
        
        # Print query breakdown for this attack
        if any(v > 0 for v in self.query_breakdown.values()):
            parts = [f"{k}={v}" for k, v in self.query_breakdown.items() if v > 0]
            print(f"  Query breakdown: {', '.join(parts)}")
        
        return stats
