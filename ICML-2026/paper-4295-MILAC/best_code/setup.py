# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='milcci',
    version='0.1.0',
    description='MILCCI: Multi-axis Interpretable Latent Component and Condition Inference',
    author='Noga Mudrik',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy<2.0',
        'scipy>=1.7',
        'scikit-learn>=1.0',
    ],
    extras_require={
        'plotting': ['matplotlib>=3.5', 'seaborn>=0.11'],
    },
)
