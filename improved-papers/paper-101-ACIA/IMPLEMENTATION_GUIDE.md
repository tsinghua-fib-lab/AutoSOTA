# ACIA Library Implementation Guide
## Complete Step-by-Step Plan

This guide walks you through converting your research code into a production-ready library.

---

## PHASE 1: Initial Setup (Day 1)

### Step 1.1: Organize Project Structure
```bash
acia/
├── acia/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── algorithms.py
│   │   ├── kernels.py
│   │   └── spaces.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── benchmarks.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── architectures.py
│   │   └── training.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── evaluation.py
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
├── examples/
│   ├── 01_colored_mnist.py
│   ├── 02_rotated_mnist.py
│   ├── 03_custom_dataset.py
│   ├── 04_camelyon17.py
│   └── 05_ball_agent.py
├── tests/
│   ├── __init__.py
│   ├── test_datasets.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_core.py
├── docs/
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   ├── api.rst
│   │   ├── tutorials.rst
│   │   └── examples.rst
│   └── Makefile
├── .github/
│   └── workflows/
│       ├── tests.yml
│       └── docs.yml
├── pyproject.toml
├── setup.py
├── README.md
├── LICENSE
├── MANIFEST.in
├── .gitignore
├── requirements.txt
└── requirements-dev.txt
```

### Step 1.2: Fix All Import Issues
**Action Items:**
1. Update all imports in your existing files to use absolute imports:
   ```python
   # OLD (don't do this)
   from causalspace import MeasurableSet
   
   # NEW (do this)
   from acia.core.spaces import MeasurableSet
   ```

2. Create __init__.py files as shown in the files I provided above

3. Fix circular dependencies:
   - Move MeasurableSet to a separate file if needed
   - Import only what's necessary in __init__.py files

### Step 1.3: Install Required Dependencies
```bash
# Create missing dependencies that your code references
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn pandas pillow cvxopt
```

**CRITICAL:** Your code references modules that don't exist:
- `causalspace` → Need to define or remove
- `causalkernel` → Need to define or remove  
- `anticausal` → Need to define or remove
- `measuretheory` → Need to define or remove

**Fix:** Either:
1. These should be internal to acia (rename imports), OR
2. Add them as external dependencies

---

## PHASE 2: Fix Code Issues (Day 1-2)

### Step 2.1: Missing Dependencies Audit
**Files to check:**
- algorithms.py line 8: `from causalkernel import CausalKernel`
- algorithms.py line 12: `from anticausal import *`
- spaces.py line 11: `from measuretheory import MeasurableSet`

**Action:** 
1. Search your codebase for where these are defined
2. Move them into acia package structure
3. Update all imports

### Step 2.2: Create Missing Base Classes
You need to create:

```python
# acia/core/measuretheory.py
class MeasurableSet:
    """Base class for measurable sets in sigma-algebra."""
    def __init__(self, data: torch.Tensor, name: str):
        self.data = data
        self.name = name
    
    def union(self, other):
        return MeasurableSet(
            torch.logical_or(self.data, other.data),
            f"({self.name} âˆª {other.name})"
        )
    
    def intersection(self, other):
        return MeasurableSet(
            torch.logical_and(self.data, other.data),
            f"({self.name} âˆ© {other.name})"
        )
    
    def complement(self):
        return MeasurableSet(
            torch.logical_not(self.data),
            f"({self.name})^c"
        )
```

### Step 2.3: Fix Inconsistencies
**Issues found:**
1. `SubSigmaAlgebra` defined in both kernels.py and spaces.py
2. `visualize_cmnist_results` in benchmarks.py should be in visualization/
3. Missing proper type hints in many functions

**Action:** Consolidate and clean up

---

## PHASE 3: Documentation (Day 2-3)

### Step 3.1: Add Docstrings
Add to EVERY public function/class:

```python
def compute_R1(self, z_H: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Compute environment independence regularizer.
    
    Enforces that high-level representations z_H should be similar
    across different environments for the same label values.
    
    Args:
        z_H: High-level representations, shape (batch_size, hidden_dim)
        y: Target labels, shape (batch_size,)
        e: Environment indicators, shape (batch_size,)
    
    Returns:
        R1 regularization term (scalar tensor)
    
    Example:
        >>> z_H = torch.randn(32, 128)
        >>> y = torch.randint(0, 10, (32,))
        >>> e = torch.randint(0, 2, (32,)).float()
        >>> R1 = optimizer.compute_R1(z_H, y, e)
    """
```

### Step 3.2: Create Sphinx Documentation
```bash
cd docs
sphinx-quickstart
# Edit conf.py to include your package
# Create tutorial pages
make html
```

### Step 3.3: Create API Documentation
Automatically generate from docstrings:
```bash
sphinx-apidoc -o docs/source/api acia
```

---

## PHASE 4: Testing (Day 3-4)

### Step 4.1: Write Unit Tests
Create tests for each module:

```python
# tests/test_datasets.py
def test_colored_mnist_shape():
    dataset = ColoredMNIST(env='e1', train=True)
    x, y, e = dataset[0]
    assert x.shape == (3, 28, 28)
    assert 0 <= y < 10

# tests/test_models.py
def test_model_forward():
    model = CausalRepresentationNetwork()
    x = torch.randn(4, 3, 28, 28)
    z_L, z_H, logits = model(x)
    assert z_L.shape == (4, 32)
    assert z_H.shape == (4, 128)
    assert logits.shape == (4, 10)
```

### Step 4.2: Run Tests
```bash
pytest tests/ -v --cov=acia --cov-report=html
```

### Step 4.3: Set Up CI/CD
Create `.github/workflows/tests.yml`:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest tests/ -v --cov=acia
```

---

## PHASE 5: Examples & Tutorials (Day 4-5)

### Step 5.1: Create Comprehensive Examples
Move to examples/ directory:
1. `01_colored_mnist.py` - Basic usage (provided above)
2. `02_rotated_mnist.py` - Rotation invariance
3. `03_custom_dataset.py` - Using your own data (provided above)
4. `04_advanced_regularizers.py` - Custom R3, R4 terms
5. `05_hyperparameter_tuning.py` - Grid search, lambda tuning

### Step 5.2: Create Jupyter Notebooks
```bash
mkdir notebooks
# Create tutorial notebooks with visualizations
```

---

## PHASE 6: Package for Distribution (Day 5)

### Step 6.1: Prepare Files
Create additional required files:

**LICENSE** (MIT recommended):
```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge...
```

**.gitignore**:
```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.venv/
venv/
ENV/
data/
*.pt
*.pth
```

**MANIFEST.in**:
```
include README.md
include LICENSE
include requirements.txt
recursive-include acia *.py
recursive-include examples *.py
recursive-include tests *.py
```

### Step 6.2: Build Package
```bash
python -m build
```

### Step 6.3: Test Installation
```bash
# Test local installation
pip install -e .

# Test that imports work
python -c "import acia; print(acia.__version__)"

# Test that examples run
python examples/01_colored_mnist.py
```

### Step 6.4: Publish to PyPI (Optional)
```bash
# Test PyPI first
python -m twine upload --repository testpypi dist/*

# Real PyPI
python -m twine upload dist/*
```

---

## PHASE 7: Polish & Release (Day 6-7)

### Step 7.1: Create Release Checklist
- [ ] All tests pass
- [ ] Documentation builds without errors
- [ ] Examples run successfully
- [ ] README is comprehensive
- [ ] LICENSE file included
- [ ] Version number updated
- [ ] CHANGELOG.md created
- [ ] GitHub repo is public
- [ ] Tagged release on GitHub

### Step 7.2: Create CHANGELOG.md
```markdown
# Changelog

## [0.1.0] - 2025-10-25
### Added
- Initial release
- ColoredMNIST, RotatedMNIST, Camelyon17, BallAgent datasets
- Causal representation network architectures
- R1 and R2 regularizers
- Measure-theoretic causal spaces
- Comprehensive documentation and examples
```

### Step 7.3: Create CONTRIBUTING.md
Guide for contributors on:
- Code style
- Testing requirements
- Documentation standards
- PR process

---

## CRITICAL FIXES NEEDED NOW

Before you can proceed, fix these issues:

### 1. Import Errors
**Problem:** Your code imports non-existent modules
```python
from causalspace import MeasurableSet  # Doesn't exist
from causalkernel import CausalKernel   # Doesn't exist
from anticausal import *                # Doesn't exist
```

**Solution A - If these are YOUR code:**
```bash
# Move them into acia package
acia/core/measuretheory.py  # Define MeasurableSet here
acia/core/kernels.py        # CausalKernel already here
```

**Solution B - If these are external:**
```bash
# Add to requirements.txt
pip install causalspace causalkernel anticausal
```

### 2. Duplicate Definitions
`SubSigmaAlgebra` is defined in both:
- `acia/core/kernels.py`
- `acia/core/spaces.py`

**Fix:** Keep one, remove the other, update imports

### 3. Missing Base Classes
You reference `MeasurableSpace` in imports but never define it.

**Fix:** Create proper base class or remove unused import

---

## RECOMMENDED TIMELINE

**Week 1: Core Infrastructure**
- Days 1-2: Fix imports, structure, dependencies
- Days 3-4: Write tests, fix bugs
- Days 5-7: Documentation, examples

**Week 2: Polish & Release**
- Days 8-10: User testing, bug fixes
- Days 11-12: Documentation polish
- Days 13-14: Release prep, PyPI upload

---

## HELPFUL COMMANDS

```bash
# Development workflow
pip install -e ".[dev]"              # Install in editable mode
pytest tests/ -v                     # Run tests
black acia/                          # Format code
isort acia/                          # Sort imports
flake8 acia/                         # Lint
mypy acia/                           # Type check
sphinx-build docs/source docs/build  # Build docs

# Package building
python -m build                      # Build package
twine check dist/*                   # Verify package
pip install dist/acia-0.1.0.tar.gz  # Test installation

# Git workflow
git add .
git commit -m "Initial library structure"
git tag v0.1.0
git push origin main --tags
```

---

## SUCCESS CRITERIA

Your library is ready when:
1. ✅ `pip install -e .` works without errors
2. ✅ All imports work: `from acia import ColoredMNIST`
3. ✅ Tests pass: `pytest tests/`
4. ✅ Examples run: `python examples/01_colored_mnist.py`
5. ✅ Documentation builds: `sphinx-build docs/source docs/build`
6. ✅ README has clear usage examples
7. ✅ GitHub repo has proper structure
8. ✅ CI/CD pipeline passes

---

## GETTING HELP

Common issues and solutions:
- **Import errors**: Check __init__.py files have proper exports
- **Module not found**: Verify package is installed (`pip list | grep acia`)
- **Tests fail**: Run with `-v` flag for details
- **Docs don't build**: Check for syntax errors in docstrings

Good luck! Start with fixing the import errors, then follow the phases in order.
