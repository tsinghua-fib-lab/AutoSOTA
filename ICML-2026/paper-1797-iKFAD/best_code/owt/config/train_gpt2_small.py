# -----------------------------------------------------------------------------
# I/O Settings
# -----------------------------------------------------------------------------
out_dir = 'out-gpt2/test'
eval_interval = 200
log_interval = 10
eval_iters = 200
eval_only = False            # <--- ADDED (Required)
always_save_checkpoint = True # <--- ADDED (Required)
ckpt_interval = 1000

init_from = 'scratch' 
load_iter = 0

# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
dataset = 'openwebtext'      # <--- ADDED (Required to find data/openwebtext/train.bin)
gradient_accumulation_steps = 40
batch_size = 12
block_size = 1024

# -----------------------------------------------------------------------------
# Model Settings
# -----------------------------------------------------------------------------
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 
bias = False
flash_attn = False           # <--- ADDED (Required for model init)

# -----------------------------------------------------------------------------
# Optimizer Settings
# -----------------------------------------------------------------------------
learning_rate = 6e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# -----------------------------------------------------------------------------
# Learning Rate Decay Scheduler
# -----------------------------------------------------------------------------
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 100000
min_lr = 3e-5

# -----------------------------------------------------------------------------
# System Settings
# -----------------------------------------------------------------------------
device = 'cuda'              # <--- ADDED (Required for setup_system)
dtype = 'float32'
compile = False              # <--- ADDED (Required for setup_model)
seed = 1337                  # <--- ADDED (Required for manual_seed)

# -----------------------------------------------------------------------------
# Metadata / Logging (Not used by Trainer, but good for reference)
# -----------------------------------------------------------------------------
comment = 'test'
save_dir = 'log_gpt2/' + comment



