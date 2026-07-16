#!/usr/bin/env python
"""
Environment Validation Script for Item Response Scaling Laws

This script checks if the environment is properly configured to run
the experiments in this repository.

Usage:
    python scripts/check_environment.py
"""

import sys
import shutil
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10.x"""
    print("Checking Python version...")
    version = sys.version_info

    if version.major == 3 and version.minor == 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (recommended)")
        return True
    elif version.major == 3 and version.minor >= 11:
        print(f"  ⚠ Python {version.major}.{version.minor}.{version.micro} (may have compatibility issues)")
        print("    Recommended: Python 3.10.x")
        print("    Some packages may not work with Python 3.11+")
        return True  # Allow but warn
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (incompatible)")
        print("    Required: Python 3.10.x")
        return False


def check_latex():
    """Check if LaTeX is installed"""
    print("\nChecking LaTeX installation...")

    if shutil.which("latex") is None:
        print("  ❌ LaTeX not found")
        print("    LaTeX is required for plot generation")
        print("\n    Install instructions:")
        print("    Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra cm-super dvipng")
        print("    macOS: brew install --cask mactex")
        print("    Windows: Install MiKTeX from https://miktex.org/download")
        return False

    try:
        result = subprocess.run(
            ["latex", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown version"
        print(f"  ✓ LaTeX found: {version_line}")
        return True
    except Exception as e:
        print(f"  ⚠ LaTeX found but could not verify version: {e}")
        return True


def check_packages():
    """Check if required Python packages are installed"""
    print("\nChecking Python packages...")

    required_packages = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "sklearn": "scikit-learn",
        "transformers": "Transformers",
        "datasets": "HuggingFace Datasets",
        "scipy": "SciPy",
        "statsmodels": "statsmodels",
        "tqdm": "tqdm",
        "joblib": "joblib",
        "tueplots": "tueplots",
    }

    all_installed = True
    for module_name, display_name in required_packages.items():
        try:
            if module_name == "sklearn":
                __import__("sklearn")
            else:
                __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} not installed")
            all_installed = False

    return all_installed


def check_pytorch_cuda():
    """Check PyTorch and CUDA availability"""
    print("\nChecking PyTorch and CUDA...")

    try:
        import torch
        print(f"  ✓ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                print(f"    - cuda:{i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  ⚠ CUDA not available (CPU-only mode)")
            print("    Experiments will run on CPU (slower but functional)")

        return True
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"  ⚠ Error checking PyTorch: {e}")
        return True


def check_huggingface():
    """Check HuggingFace Hub access"""
    print("\nChecking HuggingFace Hub access...")

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Try to check if we can access the API
        api.whoami()
        print("  ✓ HuggingFace Hub authenticated")
        return True
    except ImportError:
        print("  ❌ huggingface_hub not installed")
        return False
    except Exception:
        print("  ⚠ HuggingFace Hub not authenticated (anonymous mode)")
        print("    Public datasets will still be accessible")
        return True


def check_disk_space():
    """Check available disk space"""
    print("\nChecking disk space...")

    try:
        home = Path.home()
        cache_dir = home / ".cache" / "huggingface"

        # Get disk usage
        stat = shutil.disk_usage(home)
        free_gb = stat.free / (1024**3)

        print(f"  Available disk space: {free_gb:.1f} GB")

        if free_gb < 5:
            print("  ⚠ Low disk space (< 5 GB)")
            print("    Recommended: At least 10 GB for datasets and checkpoints")
            return True  # Warning, not error
        else:
            print(f"  ✓ Sufficient disk space")
            return True
    except Exception as e:
        print(f"  ⚠ Could not check disk space: {e}")
        return True


def print_summary(results):
    """Print summary of checks"""
    print("\n" + "="*70)
    print("ENVIRONMENT CHECK SUMMARY")
    print("="*70)

    critical_checks = [
        ("Python Version", results["python"]),
        ("LaTeX", results["latex"]),
        ("Python Packages", results["packages"]),
        ("PyTorch", results["pytorch"]),
    ]

    optional_checks = [
        ("HuggingFace Hub", results["huggingface"]),
        ("Disk Space", results["disk"]),
    ]

    print("\nCritical Requirements:")
    for name, passed in critical_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8} {name}")

    print("\nOptional/Info:")
    for name, passed in optional_checks:
        status = "✓ OK" if passed else "⚠ WARN"
        print(f"  {status:8} {name}")

    all_critical = all(passed for _, passed in critical_checks)

    print("\n" + "="*70)
    if all_critical:
        print("✅ Environment is ready! You can proceed with experiments.")
        print("\nNext steps:")
        print("  1. cd downstream && python pretrain_cat_helm_binary.py")
        print("  2. cd monkey/monkey_analysis && python testtime_calibrate.py")
        return 0
    else:
        print("❌ Environment has issues. Please fix the errors above.")
        print("\nRefer to the README.md Installation section for help:")
        print("  https://github.com/[your-repo]/README.md#installation")
        return 1


def main():
    """Main function to run all checks"""
    print("="*70)
    print("Item Response Scaling Laws - Environment Validation")
    print("="*70)

    results = {
        "python": check_python_version(),
        "latex": check_latex(),
        "packages": check_packages(),
        "pytorch": check_pytorch_cuda(),
        "huggingface": check_huggingface(),
        "disk": check_disk_space(),
    }

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
