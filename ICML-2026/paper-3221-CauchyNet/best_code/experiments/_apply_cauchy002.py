"""Apply CAUCHY-002: Replace StepLR with CosineAnnealingLR + Linear warmup."""
import re

with open("best_config_gap_filling.py") as f:
    content = f.read()

# Find train_score_one function signature and add warmup_epochs param
old_sig = "def train_score_one(name, model, train_loader, val_loader, test_data,\n                    epochs=2000, lr=0.01, imag_pen=0.0,\n                    sched_step=2000, sched_gamma=0.5, log=False):"
new_sig = "def train_score_one(name, model, train_loader, val_loader, test_data,\n                    epochs=2000, lr=0.01, imag_pen=0.0,\n                    sched_step=2000, sched_gamma=0.5, log=False,\n                    use_cosine=False, warmup_epochs=150):"
assert old_sig in content, "Could not find train_score_one signature"
content = content.replace(old_sig, new_sig)

# Replace StepLR creation with conditional cosine/step logic
old_sched = "    opt = optim.Adam(model.parameters(), lr=lr)\n    sch = optim.lr_scheduler.StepLR(opt, step_size=sched_step, gamma=sched_gamma)"
new_sched = """    opt = optim.Adam(model.parameters(), lr=lr)
    if use_cosine:
        warmup = optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs - warmup_epochs, eta_min=lr * 1e-3)
        sch = optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        sch = optim.lr_scheduler.StepLR(opt, step_size=sched_step, gamma=sched_gamma)"""
assert old_sched in content, "Could not find scheduler creation"
content = content.replace(old_sched, new_sched)

with open("best_config_gap_filling.py", "w") as f:
    f.write(content)
print("Applied CAUCHY-002: cosine annealing + warmup option added")
