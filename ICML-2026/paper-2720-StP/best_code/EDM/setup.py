from setuptools import setup, find_packages

setup(
    name='EN_diffusion',
    version='1.0.0',
    url=None,
    author='Author Name',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1']
)