"""Apply CAUCHY-005: Curriculum training with gap-adjacent points in Phase 2."""
with open("best_config_gap_filling.py") as f:
    content = f.read()

# 1. Add build_curriculum_data function after build_data
old_after_build = """    return (train_X.unsqueeze(-1), f(train_X).unsqueeze(-1),
            val_X.unsqueeze(-1),   f(val_X).unsqueeze(-1),
            test_X.unsqueeze(-1),  f(test_X).unsqueeze(-1))"""
assert old_after_build in content, "Could not find build_data return"

new_after_build = """    return (train_X.unsqueeze(-1), f(train_X).unsqueeze(-1),
            val_X.unsqueeze(-1),   f(val_X).unsqueeze(-1),
            test_X.unsqueeze(-1),  f(test_X).unsqueeze(-1))


def build_gap_adjacent_points(seed=10, n_points=50):
    """Generate points in gap-adjacent region (distance in [0.15, 0.25] from turning points).
    These are outside the test region (<0.15) but near gaps for curriculum fine-tuning."""
    torch.manual_seed(seed); np.random.seed(seed)
    X1 = torch.linspace(-2, 2, 2000)
    with torch.no_grad(): dY1 = df(X1)
    turning_pts = X1[torch.abs(dY1) < 0.15]
    out = []
    while len(out) < n_points:
        samp = torch.normal(0., 1., size=(1,))*2.
        samp = torch.clamp(samp, -2, 2)
        min_dist = torch.min(torch.abs(turning_pts - samp))
        # Accept points in [0.15, 0.25] distance range
        if 0.15 <= min_dist < 0.25:
            out.append(samp)
    gap_X = torch.cat(out)
    return gap_X.unsqueeze(-1), f(gap_X).unsqueeze(-1)"""
content = content.replace(old_after_build, new_after_build)

# 2. Modify train_score_one signature and body for curriculum
old_sig = "def train_score_one(name, model, train_loader, val_loader, test_data,\n                    epochs=2000, lr=0.01, imag_pen=0.0,\n                    sched_step=2000, sched_gamma=0.5, log=False,\n                    use_cosine=False, warmup_epochs=150):"
new_sig = "def train_score_one(name, model, train_loader, val_loader, test_data,\n                    epochs=2000, lr=0.01, imag_pen=0.0,\n                    sched_step=2000, sched_gamma=0.5, log=False,\n                    use_cosine=False, warmup_epochs=150,\n                    curriculum_loader=None, phase1_epochs=1500):"
assert old_sig in content, "Could not find train_score_one sig"
content = content.replace(old_sig, new_sig)

# 3. Add curriculum phase switch in training loop
# Find the main training loop and add phase switching
# After sch.step(), check if we need to switch to curriculum data
old_sch_step = "        sch.step()\n        train_curve.append"
assert old_sch_step in content, "Could not find sch.step()"

new_sch_step = """        sch.step()
        # Curriculum: switch to gap-adjacent data at phase1_epochs
        if curriculum_loader is not None and ep == phase1_epochs - 1:
            train_loader = curriculum_loader
        train_curve.append"""
content = content.replace(old_sch_step, new_sch_step)

with open("best_config_gap_filling.py", "w") as f:
    f.write(content)
print("Applied CAUCHY-005: curriculum training with gap-adjacent points")
