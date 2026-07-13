#!/usr/bin/env python3
"""
Patch train_NAVAR.py with three improvements:
1. Early stopping with best-model checkpointing (Idea #5)
2. Cosine annealing LR schedule with linear warmup (Idea #6)
3. Gradient clipping (Idea #9)
"""
with open("/repo/train_NAVAR.py", "r") as f:
    content = f.read()

# === 1. Add gradient clipping before optimizer.step() ===
old_clip = "            # Zero gradients, perform a backward pass, and update the weights.\n            optimizer.zero_grad()\n            loss.backward()\n            optimizer.step()"
new_clip = "            # Zero gradients, perform a backward pass, and update the weights.\n            optimizer.zero_grad()\n            loss.backward()\n            # Gradient clipping for stability (Idea #9)\n            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)\n            optimizer.step()"
content = content.replace(old_clip, new_clip)

# === 2. Add LR scheduler after optimizer creation ===
old_lr = "    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)"
new_lr = """    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Cosine annealing LR with linear warmup (Idea #6)
    warmup_epochs = max(1, epochs // 20)  # 5% warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )"""
content = content.replace(old_lr, new_lr)

# === 3. Add early stopping and best-model tracking ===
old_train_loop_start = "    # start of training loop\n    batch_counter = 0\n    for t in range(1, epochs +1):"
new_train_loop_start = """    # Early stopping setup (Idea #5)
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 4  # stop after 4 validation checks without improvement
    min_epochs = 600  # must train at least this many epochs

    # start of training loop
    batch_counter = 0
    for t in range(1, epochs +1):"""
content = content.replace(old_train_loop_start, new_train_loop_start)

# Add scheduler step after each epoch
old_epoch_end = "            optimizer.step()\n\n        # every 'check_every' epochs we calculate and print the validation loss"
new_epoch_end = """            optimizer.step()
            scheduler.step()

        # every 'check_every' epochs we calculate and print the validation loss"""
content = content.replace(old_epoch_end, new_epoch_end)

# Add early stopping logic after validation loss print
old_val_print = """            model.train()

            print(f'iteration {t}. Loss: {total_loss/batch_counter}  Val loss: {loss_val}')"""
new_val_print = """            model.train()

            print(f'iteration {t}. Loss: {total_loss/batch_counter}  Val loss: {loss_val}')
            # Early stopping check (Idea #5)
            if val_proportion > 0.0 and t >= min_epochs:
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'  Early stopping at epoch {t} (no improvement for {patience} checks)')
                        break"""
content = content.replace(old_val_print, new_val_print)

# Add best model restoration before final eval
old_final_eval = "    # use the trained model to calculate the causal scores\n    model.eval()\n    y_pred, contributions = model(X_train)"
new_final_eval = """    # Restore best model if early stopping was active (Idea #5)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'  Restored best model (val_loss={best_val_loss:.6f})')

    # use the trained model to calculate the causal scores
    model.eval()
    y_pred, contributions = model(X_train)"""
content = content.replace(old_final_eval, new_final_eval)

with open("/repo/train_NAVAR.py", "w") as f:
    f.write(content)
print("train_NAVAR.py patched with all three improvements")
