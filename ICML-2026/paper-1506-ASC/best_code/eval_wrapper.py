#!/usr/bin/env python3
"""Fast eval wrapper for SOTA optimization. Runs run_separable_exp and outputs JSON metrics."""
import subprocess, sys, json, re, os

def run_eval(k=3, cost_scaling=0.5, lr=0.006, weight_decay=0.001, epochs=500, seed=1,
             adam_weight_decay=None, reg_classifier=None, reg_auxiliary=None,
             dev=None, tau_start=None, tau_end=None):
    """Run a single experiment and return parsed metrics."""
    cmd = [sys.executable, "/repo/run_separable_exp.py",
           "--k", str(k), "--cost-scaling", str(cost_scaling),
           "--lr", str(lr), "--epochs", str(epochs), "--seed", str(seed)]
    
    # Build env with overrides
    env = os.environ.copy()
    if adam_weight_decay is not None:
        env["SOTA_ADAM_WD"] = str(adam_weight_decay)
    if reg_classifier is not None:
        env["SOTA_REG_CLASSIFIER"] = str(reg_classifier)
    if reg_auxiliary is not None:
        env["SOTA_REG_AUXILIARY"] = str(reg_auxiliary)
    if dev is not None:
        env["SOTA_DEV"] = str(dev)
    if tau_start is not None:
        env["SOTA_TAU_START"] = str(tau_start)
    if tau_end is not None:
        env["SOTA_TAU_END"] = str(tau_end)
    
    if weight_decay != 0.001:
        cmd.extend(["--weight-decay", str(weight_decay)])
    
    print(f"Running: {
