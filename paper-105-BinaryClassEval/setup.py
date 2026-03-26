from setuptools import setup, find_packages

setup(
    name="epamnb",
    version="0.1.0",
    description="Visualizing prior-adjusted maximum net benefit curves with subgroup analysis",
    author="Gerardo Flores",
    author_email="gfm@mit.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
    ],
    python_requires=">=3.8",
)