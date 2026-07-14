"""Test the optimized implementation on 3 splits."""
import sys, os
sys.path.insert(0, "/repo")
os.chdir("/repo")

# Import and override N_SPLITS
import run_har_optimized as runner
runner.N_SPLITS = 3  # Quick test with 3 splits

df = runner.main()
