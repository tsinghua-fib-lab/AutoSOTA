# IDEA-01: Gumbel-Softmax temperature annealing
# Adds tau_schedule to DynamicGlobalMultiLineFragsMoE
import re

# 1. Modify simple_snn.py: add tau_schedule to dynamic_frags config
with open("/repo/simple_snn.py", "r") as f:
    content = f.read()

# Add tau_schedule=True after gumbel_tau=1.0
old = "        gumbel_tau=1.0,\n        gumbel_hard=True,"
new = "        gumbel_tau=1.0,\n        tau_schedule=True,\n        tau_start=5.0,\n        tau_end=0.5,\n        tau_anneal_epochs=30,\n        gumbel_hard=True,"
content = content.replace(old, new)

with open("/repo/simple_snn.py", "w") as f:
    f.write(content)

# 2. Modify learnable_fragmentation.py: add tau_schedule support
with open("/repo/learnable_fragmentation.py", "r") as f:
    content = f.read()

# Add tau_schedule params to __init__
old_init = "        init_logit_bias: float = 4.0,\n    ) -> None:"
new_init = "        init_logit_bias: float = 4.0,\n        # tau annealing\n        tau_schedule: bool = False,\n        tau_start: float = 5.0,\n        tau_end: float = 0.5,\n        tau_anneal_epochs: int = 30,\n    ) -> None:"
content = content.replace(old_init, new_init)

# Store tau schedule params
old_super = "        self.n_angles = int(n_angles)"
new_super = "        self.tau_schedule = tau_schedule\n        self.tau_start = tau_start\n        self.tau_end = tau_end\n        self.tau_anneal_epochs = tau_anneal_epochs\n        self.n_angles = int(n_angles)"
content = content.replace(old_super, new_super)

# Modify _select_distribution to use scheduled tau
old_tau = "            p = F.gumbel_softmax(self.step_logits, tau=self.gumbel_tau, hard=False, dim=0)"
new_tau = "            # Compute scheduled tau if enabled\n            if self.tau_schedule and self.training:\n                # epoch estimated from iter count: 600 iters/epoch\n                est_epoch = max(0, (int(self._iter.item()) - self.warmup_iters) / 600)\n                if est_epoch <= 0:\n                    current_tau = self.tau_start\n                elif est_epoch >= self.tau_anneal_epochs:\n                    current_tau = self.tau_end\n                else:\n                    import math\n                    progress = est_epoch / self.tau_anneal_epochs\n                    current_tau = self.tau_end + (self.tau_start - self.tau_end) * (1 + math.cos(math.pi * progress)) / 2\n            else:\n                current_tau = self.gumbel_tau\n            p = F.gumbel_softmax(self.step_logits, tau=current_tau, hard=False, dim=0)"
content = content.replace(old_tau, new_tau)

with open("/repo/learnable_fragmentation.py", "w") as f:
    f.write(content)

print("IDEA-01 patch applied")
