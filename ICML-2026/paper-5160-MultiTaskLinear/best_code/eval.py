#!/usr/bin/env python3
"""Evaluation script for paper 5160: Multi-task Linear Regression HAR experiment.

Reproduces the real-data HAR experiment from Section 7.2 / Table 1.
Protocol: 30 splits, 5-fold CV, q in {0.05,...,0.50}, standing-vs-others.

Uses reduced CV iterations (maxiter_cv=50) for speed while maintaining
full convergence (maxiter_full=200) for the final refit.
"""
import sys, time
sys.path.insert(0, "/repo")
from run_har_optimized import main

if __name__ == "__main__":
    t0 = time.time()
    df = main()
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")
