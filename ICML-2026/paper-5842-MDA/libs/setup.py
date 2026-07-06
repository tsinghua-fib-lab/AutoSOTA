from setuptools import setup, find_packages

setup(
    name="transformer-lens",
    version="2.15.0.post0",
    description="TransformerLens (patched for MDA)",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "accelerate",
        "beartype>=0.14.0",
        "better-abc>=0.0.3",
        "datasets>=2.7.1",
        "einops>=0.6.0",
        "fancy-einsum>=0.0.3",
        "jaxtyping>=0.2.11",
        "numpy>=1.24",
        "pandas>=1.1.5",
        "rich>=12.6.0",
        "sentencepiece",
        "torch>=2.2",
        "tqdm>=4.64.1",
        "transformers>=4.43",
        "transformers-stream-generator>=0.0.5,<0.0.6",
        "typeguard>=4.2,<5.0",
        "typing-extensions",
        "wandb>=0.13.5",
    ],
)
