"""Apply CAUCHY-003: L1 regularization on real part of hidden activations."""
with open("best_config_gap_filling.py") as f:
    content = f.read()

# Add sparse_penalty parameter to train_score_one
old_sig = "def train_score_one(name, model, train_loader, val_loader, test_data,\n                    epochs=2000, lr=0.01, imag_pen=0.0,\n                    sched_step=2000, sched_gamma=0.5, log=False,\n                    use_cosine=False, warmup_epochs=150):"
new_sig = "def train_score_one(name, model, train_loader, val_loader, test_data,\n                    epochs=2000, lr=0.01, imag_pen=0.0,\n                    sched_step=2000, sched_gamma=0.5, log=False,\n                    use_cosine=False, warmup_epochs=150,\n                    sparse_penalty=0.0):"
assert old_sig in content, "Could not find train_score_one sig"
content = content.replace(old_sig, new_sig)

# We need to capture "activated" in the loss computation for sparse penalty
# Find the training loop body where loss is computed and add sparse penalty
old_clip = "            loss.backward()\n            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)"
assert old_clip in content, "Could not find loss.backward + clip"

# We need to insert sparse penalty computation BEFORE loss.backward()
# The loss is computed above this. We need to add sparse_penalty to the loss.
# Since CauchyNet returns a tuple, we need to hook into the imag_pen branch.

# Actually, the cleanest approach: add sparse_penalty after loss computation but before backward.
# But wait, we also need the "activated" tensor, which is only accessible inside CauchyNet.forward().
# Solution: modify CauchyNet.forward() to also return the hidden activations when needed.

# Alternative: add sparse penalty directly in CauchyNet.forward() as an attribute
# Simplest approach: pass a flag to forward() and store activations as model attribute

# Let me modify CauchyNet.forward to optionally store activations
old_forward = "        out_c = torch.matmul(activated, self.lambda_)"
assert old_forward in content, "Could not find forward matmul"

# Add activation storage
new_forward_body = """        self._last_activated = activated  # store for sparse penalty
        out_c = torch.matmul(activated, self.lambda_)"""
content = content.replace(old_forward, new_forward_body)

# Now add the sparse penalty to the loss computation in train_score_one
# Find: loss = crit(yr, yb) + imag_pen*crit(yi, torch.zeros_like(yi))
old_loss = "                loss = crit(yr, yb) + imag_pen*crit(yi, torch.zeros_like(yi))"
new_loss = """                loss = crit(yr, yb) + imag_pen*crit(yi, torch.zeros_like(yi))
                if sparse_penalty > 0 and hasattr(model, "_last_activated"):
                    loss = loss + sparse_penalty * torch.norm(model._last_activated.real, p=1)"""
assert old_loss in content, "Could not find loss line"
content = content.replace(old_loss, new_loss)

with open("best_config_gap_filling.py", "w") as f:
    f.write(content)
print("Applied CAUCHY-003: L1 sparse penalty on hidden activations")
