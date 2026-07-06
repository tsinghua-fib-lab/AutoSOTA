from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="fair_ot",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"src": "src"},
    install_requires=requirements,
)
