import os
import re
from setuptools import setup

with open(os.path.join("visionts", "__init__.py"), "r") as f:
    content = f.read()
version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M).group(1)
author = re.search(r"^__author__ = ['\"]([^'\"]*)['\"]", content, re.M).group(1)

setup(
    name='visionts',
    version=version,
    author=author,
    author_email='chenmx@zju.edu.cn',
    description='Using a visual MAE for time series forecasting.',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Keytoyze/VisionTS',
    packages=['visionts'],
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "torchvision",
        "einops",
        "numpy",
        "timm",
        "pandas",
        "huggingface_hub",
    ],
)