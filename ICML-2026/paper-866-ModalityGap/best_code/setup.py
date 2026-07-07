import setuptools

setuptools.setup(
    name='online-learning-ood',
    version='1.0.0',
    description='Online Learning for OOD Detection with CLIP',
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.13.1',
        'torchvision>=0.13',
        'numpy',
        'scikit-learn',
        'tqdm',
        'pyyaml>=5.4.1',
        'Pillow',
        # CLIP tokeniser dependencies.
        'ftfy',
        'regex',
    ],
)
