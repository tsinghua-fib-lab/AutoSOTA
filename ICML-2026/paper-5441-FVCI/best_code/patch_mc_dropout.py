#!/usr/bin/env python3
"""Patch reproduce_threshold.py with MC Dropout at inference (Idea #2)."""
with open("/repo/reproduce_threshold.py", "r") as f:
    content = f.read()

# 1. Replace model_predict to support MC dropout
old_predict = """def model_predict(model, Xwin_np, batch_size=512):
    \"\"\"Run model forward in eval mode. Returns predictions (W, N).\"\"\"
    model.eval()
    device = next(model.parameters()).device
    preds_out = []
    with torch.no_grad():
        for i in range(0, len(Xwin_np), batch_size):
            xb = torch.tensor(Xwin_np[i:i+batch_size], dtype=torch.float32, device=device)
            out = model(xb)
            pred = (out[0] if isinstance(out, (tuple, list)) else out).detach().cpu().numpy()
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred[:, :, 0]
            preds_out.append(pred)
    return np.vstack(preds_out)"""

new_predict = """def model_predict(model, Xwin_np, batch_size=512, mc_dropout=False, num_mc_passes=50):
    \"\"\"Run model forward. If mc_dropout=True, keeps dropout active and averages
    over multiple stochastic forward passes (MC Dropout for Bayesian averaging).
    Returns predictions (W, N).\"\"\"
    device = next(model.parameters()).device
    n_samples = len(Xwin_np)

    if mc_dropout:
        model.train()  # keep dropout active
        accum = np.zeros((n_samples, model.num_nodes), dtype=np.float64)
        for _ in range(num_mc_passes):
            preds_out = []
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    xb = torch.tensor(Xwin_np[i:i+batch_size], dtype=torch.float32, device=device)
                    out = model(xb)
                    pred = (out[0] if isinstance(out, (tuple, list)) else out).detach().cpu().numpy()
                    if pred.ndim == 3 and pred.shape[-1] == 1:
                        pred = pred[:, :, 0]
                    preds_out.append(pred)
            accum += np.vstack(preds_out)
        return (accum / num_mc_passes).astype(np.float32)
    else:
        model.eval()
        preds_out = []
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                xb = torch.tensor(Xwin_np[i:i+batch_size], dtype=torch.float32, device=device)
                out = model(xb)
                pred = (out[0] if isinstance(out, (tuple, list)) else out).detach().cpu().numpy()
                if pred.ndim == 3 and pred.shape[-1] == 1:
                    pred = pred[:, :, 0]
                preds_out.append(pred)
        return np.vstack(preds_out)"""

content = content.replace(old_predict, new_predict)

# 2. Update estimate_ice_synthetic to pass mc_dropout param
old_ice = """def estimate_ice_synthetic(model, data_np, src_idx=0, tgt_idx=1,
                            maxlags=1, grid=None, grid_q=(0.02, 0.98), grid_n=81):"""

new_ice = """def estimate_ice_synthetic(model, data_np, src_idx=0, tgt_idx=1,
                            maxlags=1, grid=None, grid_q=(0.02, 0.98), grid_n=81,
                            mc_dropout=False, num_mc_passes=50):"""

content = content.replace(old_ice, new_ice)

# 3. Update model_predict calls inside estimate_ice_synthetic to pass mc params
old_pred_base = "    pred_base = model_predict(model, Xwin)[:, tgt_idx]"
new_pred_base = "    pred_base = model_predict(model, Xwin, mc_dropout=mc_dropout, num_mc_passes=num_mc_passes)[:, tgt_idx]"
content = content.replace(old_pred_base, new_pred_base)

old_pred_mod = "(model_predict(model, Xmod)[:, tgt_idx] - pred_base).mean()"
new_pred_mod = "(model_predict(model, Xmod, mc_dropout=mc_dropout, num_mc_passes=num_mc_passes)[:, tgt_idx] - pred_base).mean()"
content = content.replace(old_pred_mod, new_pred_mod)

# 4. Add mc_dropout params to CFG
old_cfg_end = """        split_timeseries=None,
    )"""
new_cfg_end = """        split_timeseries=None,
        mc_dropout=False,
        num_mc_passes=50,
    )"""
content = content.replace(old_cfg_end, new_cfg_end)

# 5. Pass mc_dropout to estimate_ice_synthetic call
old_ice_call = """        _, g_hat = estimate_ice_synthetic(
            model, data_ice,
            src_idx=0, tgt_idx=1,
            maxlags=CFG['maxlags'],
            grid=grid,
            grid_n=GRID_N,
        )"""

new_ice_call = """        _, g_hat = estimate_ice_synthetic(
            model, data_ice,
            src_idx=0, tgt_idx=1,
            maxlags=CFG['maxlags'],
            grid=grid,
            grid_n=GRID_N,
            mc_dropout=CFG['mc_dropout'],
            num_mc_passes=CFG['num_mc_passes'],
        )"""

content = content.replace(old_ice_call, new_ice_call)

with open("/repo/reproduce_threshold.py", "w") as f:
    f.write(content)
print("MC Dropout support added to reproduce_threshold.py")
