from setuptools import setup

requirements = [
    "numpy>=1.16",
    "scipy>=1.3",
    "torch>=2.1.1",
    "gpytorch>=1.11",
    "botorch<0.10",
    "wandb>=0.16",
    "matplotlib>=3.7",
    "tqdm>=4.0",
    "notebook>=6.0",
    "ipywidgets>=8.1.1",
    "scikit-learn>=1.1",
    "pandas>=2.2",
    "openml>=0.14.2"
]

setup(
    name="pandora_automl",
    version="1.0",
    description="Cost-aware Stopping for Bayesian Optimization",
    author="Qian Xie",
    python_requires='>=3.9',
    packages=["pandora_automl"],
    install_requires=requirements,
    extras_require={
        "lunar_lander": ["gymnasium>=0.29.0", "box2d>=2.3.10", "pygame>=2.0.0"],
        "robot_pushing": ["pygame>=2.0.0", "box2d>=2.3.10"],
    }
)