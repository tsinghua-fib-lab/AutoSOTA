from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gamba",
    version="0.1",
    description="Python package for generation of DNA sequences and corresponding evolutionary information",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pybigwig",
        "pandas",
        "pyfaidx",
        "matplotlib",
        "scipy",
        "wandb",
        "biopython",
        "transformers",
        "blosum",
        "torchaudio",
        "torchvision",
        "fair-esm",
        "seaborn",
        "cudatoolkit",
        "cuda-nvcc",
    ],
)


# name: gamba
# channels:
#   - defaults
# dependencies:
#   - python==3.12.2
#   - pybigwig
#   - pandas
#   - bioconda::pyfaidx
#   - matplotlib
#   - scipy==1.13.1
#   - wandb
#   - biopython=1.83
#   - transformers=4.40.2
#   - blosum=2.0.3
#   - torchaudio
#   - torchvision
#   - fair-esm=2.0.0
#   - seaborn
#   - cudatoolkit==11.1.1
#   - cuda-nvcc
