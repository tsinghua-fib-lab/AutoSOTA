from setuptools import setup, find_packages
import io
import os

here = os.path.abspath(os.path.dirname(__file__))
long_description = """
gconvex: Finitely convex parametrization of generalized convex functions
and experimental optimal_transport for Monge map / optimal transport comparisons.
"""
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except Exception:
    pass

setup(
    name="gconvex",
    version="0.1.0",
    description="Finitely convex parametrization and OT optimal_transport",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Moeen Nehzati",
    url="https://github.com/MoeenNehzati/gconvex",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        # Core numeric / ML (torch>=2.0 needed for torch.compile)
        "torch>=2.0.0,<3.0",
        "numpy>=1.19.0,<2.0",
        # Visualization & data
        "matplotlib>=3.3.0,<4.0",
        "pandas>=1.1.0,<2.0",
        # Logging / notebook utilities
        "rich>=10.0.0",
        "ipython>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-xdist>=3.0.0",  # Parallel test execution
            "pytest-timeout>=2.0.0",  # Test timeouts
            "flake8",
        ],
        "wandb": ["wandb"],
    },
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
