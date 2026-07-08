from setuptools import setup, find_packages

setup(
    name="csr",
    version="1.0.0",
    description="CSR: CLIP Spectral Robustness — A frequency-domain adversarial defense for CLIP",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",
        "torchvision>=0.14",
        "transformers>=4.25",
        "open_clip_torch>=2.0",
        "pandas",
        "numpy",
        "scipy",
        "Pillow",
        "tqdm",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "matplotlib",
            "seaborn",
            "jupyter",
            "tabulate",
        ],
    },
)
