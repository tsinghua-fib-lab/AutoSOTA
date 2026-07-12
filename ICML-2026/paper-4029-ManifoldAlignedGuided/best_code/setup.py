from setuptools import find_packages
from setuptools import setup

setup(
    name="magig",
    version="1.0.0",
    description="Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution (ICML 2026).",
    url="https://github.com/leekwoon/ma-gig",
    license="MIT",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
)
