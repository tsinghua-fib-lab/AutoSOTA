from setuptools import setup, find_packages

setup(
    name='olla_sampling',
    version='0.1.0',
    description='A collection of algorithms for sampling from constrained distributions, including OLLA, CHMC, and CGHMC.',
    author='Kijung Jeon',
    packages=find_packages(),
    python_requires='>=3.8'
)
