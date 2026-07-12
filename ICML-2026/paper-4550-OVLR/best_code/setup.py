from setuptools import setup, find_packages

setup(
    name='ovlr',
    version='1.0.0',
    description='Output-Level Variance-Reduced Likelihood Ratio Gradient Estimation',
    author='OVLR Authors',
    url='https://github.com/ovlr/ovlr',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
        'torchvision>=0.12.0',
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license='MIT',
)
