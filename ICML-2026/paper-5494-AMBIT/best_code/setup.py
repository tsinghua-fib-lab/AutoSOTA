from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cobet",
    version="0.1.0",
    author="Your Name",
    author_email="your@email.com",
    description="Adaptive Multiscale Binary Expansion Tests for Independence (CoBET, dCoBET, wa-dCoBET)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/cobet",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        "matplotlib>=3.4",
        "openpyxl>=3.0",
    ],
    extras_require={
        "baselines": ["hyppo>=0.3"],
        "dev": ["pytest", "jupyter", "xlsxwriter"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Statistics",
        "Intended Audience :: Science/Research",
    ],
    keywords="independence test copula binary expansion nonparametric statistics",
)
