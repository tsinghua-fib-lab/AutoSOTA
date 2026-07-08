"""
KTO 控制器模块

策略输出：
- target_class: int, 目标类别/bin (0 ~ num_targets-1)
- template_id: int, mask 模板 ID (0=explore, 1=conservative)
- explore_level: float, 探索程度 (0~1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import NamedTuple
from config import KTOConfig


class Action(NamedTuple):
    """策略动作"""
    target_class: int       # 目标类别/bin (0 ~ num_targets-1)
    template_id: int        # mask 模板 ID (0=explore, 1=conservative)
    explore_level: float    # 探索程度 (0~1)


class HeuristicRefPolicy:
    """固定的启发式参考策略（安全锚：补正缺口 + 自适应保守）"""
    
    def __init__(self, num_targets: int, num_templates: int):
        self.num_targets = num_targets
        self.num_templates = num_templates
    
    def log_prob(self, state: torch.Tensor, action: Action) -> torch.Tensor:
        # 1. 目标：只对正deficit做softmax（负deficit给很小概率）
        deficit = state[:, :self.num_targets]
        deficit_pos = torch.clamp(deficit, min=0) + 1e-6
        target_lp = F.log_softmax(deficit_pos * 5, dim=-1)[:, action.target_class]
        
        # 2. 模板：explore通过率低→更偏conservative
        gate_rates = state[:, 2*self.num_targets : 2*self.num_targets + self.num_templates]
        explore_gate = gate_rates[:, 0]
        p_conservative = torch.sigmoid(2 * (0.5 - explore_gate))
        template_probs = torch.stack([1 - p_conservative, p_conservative], dim=-1)
        template_lp = torch.log(template_probs[:, action.template_id] + 1e-8)
        
        # 3. explore_level：偏保守(0.4)，方差小(0.15)
        explore_lp = -0.5 * ((action.explore_level - 0.4) / 0.15) ** 2
        
        return target_lp + template_lp + explore_lp


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(
        self,
        state_dim: int,
        num_targets: int,
        num_templates: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.num_targets = num_targets
        self.num_templates = num_templates
        
        layers = []
        input_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)
        
        self.target_head = nn.Linear(hidden_dim, num_targets)
        self.template_head = nn.Linear(hidden_dim, num_templates)
        self.explore_mean = nn.Linear(hidden_dim, 1)
        self.explore_logstd = nn.Parameter(torch.zeros(1))
    
    def forward(self, state: torch.Tensor) -> dict:
        features = self.shared_net(state)
        return {
            "target_logits": self.target_head(features),
            "template_logits": self.template_head(features),
            "explore_mean": torch.sigmoid(self.explore_mean(features)),
            "explore_std": torch.exp(self.explore_logstd).expand(state.shape[0], 1),
        }
    
    def sample_action(self, state: torch.Tensor) -> Action:
        out = self.forward(state)
        target = torch.distributions.Categorical(logits=out["target_logits"]).sample().item()
        template_id = torch.distributions.Categorical(logits=out["template_logits"]).sample().item()
        explore = (out["explore_mean"] + torch.randn_like(out["explore_mean"]) * out["explore_std"]).clamp(0, 1)
        return Action(target, template_id, explore.item())
    
    def log_prob(self, state: torch.Tensor, action: Action) -> torch.Tensor:
        out = self.forward(state)
        target_lp = F.log_softmax(out["target_logits"], dim=-1)[:, action.target_class]
        template_lp = F.log_softmax(out["template_logits"], dim=-1)[:, action.template_id]
        explore_lp = -0.5 * ((action.explore_level - out["explore_mean"]) / (out["explore_std"] + 1e-8)) ** 2
        return target_lp + template_lp + explore_lp.squeeze(-1)


class KTOAgent:
    """KTO 智能体（单样本二元更新）"""
    
    def __init__(
        self,
        state_dim: int,
        num_targets: int,
        num_templates: int,
        config: KTOConfig,
        device: str = "cpu"
    ):
        self.device = device
        self.config = config
        self.num_targets = num_targets
        
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            num_targets=num_targets,
            num_templates=num_templates,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        ).to(device)
        
        # 固定的启发式参考策略
        self.ref_policy = HeuristicRefPolicy(num_targets, num_templates)
        
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=config.policy_lr)
        
        # z_0: KL参考点（EMA更新）
        self.z_0 = 0.0
        self.z_0_ema = 0.95
        
        # 不平衡统计
        self.n_desirable = 0
        self.n_undesirable = 0
        self.n_skip = 0
        self.step_count = 0
        self.ig_abs_buffer = deque(maxlen=config.ig_window_size)
        self.ig_ema = 0.0
        self.ig_ema_beta = 0.99
    
    def select_action(self, state: np.ndarray) -> Action:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy.sample_action(state_t)
    
    def update(self, state: np.ndarray, action: Action, ig_estimate: float) -> float:
        """单样本KTO更新"""
        if (self.n_desirable + self.n_undesirable + self.n_skip) == 0:
            self.ig_ema = ig_estimate
        ig_signed = ig_estimate - self.ig_ema
        self.ig_ema = self.ig_ema_beta * self.ig_ema + (1 - self.ig_ema_beta) * ig_estimate
        if len(self.ig_abs_buffer) >= self.config.ig_min_buf:
            tau = float(np.quantile(self.ig_abs_buffer, self.config.ig_quantile_q))
            tau = max(tau, 0.02)
        else:
            tau = self.config.tau_warmup
        self.ig_abs_buffer.append(abs(ig_signed))
        
        # 二元标签
        if ig_signed > tau:
            is_desirable = True
            self.n_desirable += 1
        elif ig_signed < -tau:
            is_desirable = False
            self.n_undesirable += 1
        else:
            self.n_skip += 1
            return 0.0  # skip
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        log_prob = self.policy.log_prob(state_t, action)
        ref_log_prob = self.ref_policy.log_prob(state_t, action)
        r_theta = log_prob - ref_log_prob
        
        # 更新z_0
        with torch.no_grad():
            r_val = r_theta.item()
        self.z_0 = self.z_0_ema * self.z_0 + (1 - self.z_0_ema) * r_val if self.step_count > 0 else r_val
        
        # λD/λU动态调整
        n_total = self.n_desirable + self.n_undesirable
        if n_total > 10:
            ratio_D = self.n_desirable / n_total
            ratio_U = self.n_undesirable / n_total
            lambda_D = self.config.lambda_D / max(ratio_D, 0.1)
            lambda_U = self.config.lambda_U / max(ratio_U, 0.1)
            
        else:
            lambda_D, lambda_U = self.config.lambda_D, self.config.lambda_U
        
        # KTO loss
        if is_desirable:
            loss = lambda_D * (1 - torch.sigmoid(self.config.beta_kto * (r_theta - self.z_0)))
        else:
            loss = lambda_U * torch.sigmoid(self.config.beta_kto * (r_theta - self.z_0))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        self.step_count += 1
        return loss.item()
