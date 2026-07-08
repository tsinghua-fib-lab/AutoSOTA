# config/train_gpt_xs.py
# -----------------------------------------------------------------------------
# I/O Settings
# -----------------------------------------------------------------------------
out_dir = 'out-gpt2/test'
eval_interval = 200
log_interval = 10
eval_iters = 200
eval_only = False 
always_save_checkpoint = True
ckpt_interval = 1000

init_from = 'scratch' 
load_iter = 0

# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
dataset = 'openwebtext'
gradient_accumulation_steps = 16
batch_size = 16
block_size = 1024

# -----------------------------------------------------------------------------
# Model Settings
# -----------------------------------------------------------------------------
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.0 
bias = False # no biases in linear and layer norm
flash_attn = True

# -----------------------------------------------------------------------------
# Optimizer Settings
# -----------------------------------------------------------------------------
learning_rate = 5e-4
max_iters = 5000
weight_decay = 0
grad_clip = 1.0

# -----------------------------------------------------------------------------
# System Settings
# -----------------------------------------------------------------------------
device = 'cuda'
dtype = 'float16'
compile = False
seed = 1337

# -----------------------------------------------------------------------------
# Metadata / Logging (Not used by Trainer, but good for reference)
# -----------------------------------------------------------------------------
comment = 'test'
save_dir = 'log_gpt2/' + comment

