# Apply IDEA-03: Cosine annealing LR schedule with warmup
with open('/repo/simple_snn.py', 'r') as f:
    content = f.read()

# Add import
content = content.replace(
    'import numpy as np',
    'import numpy as np\nfrom torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR'
)

# Replace scheduler at all 3 locations
old = 'scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)'
new = ('warmup_epochs_lr = 5\n'
       'warmup_lr = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs_lr)\n'
       'cosine_lr = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs_lr, eta_min=1e-6)\n'
       'scheduler = SequentialLR(optimizer, schedulers=[warmup_lr, cosine_lr], milestones=[warmup_epochs_lr])')

lines = content.split('\n')
new_lines = []
for line in lines:
    if old in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(indent + new.replace('\n', '\n' + indent))
    else:
        new_lines.append(line)
content = '\n'.join(new_lines)

with open('/repo/simple_snn.py', 'w') as f:
    f.write(content)

print('Cosine LR schedule applied')
