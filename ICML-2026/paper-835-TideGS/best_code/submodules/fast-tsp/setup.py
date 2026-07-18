from __future__ import annotations

import pathlib
import re
import sys

try:
    from skbuild import setup
except ImportError:
    print(
        'Please update pip, you need pip 10 or greater,\n'
        ' or you need to install the PEP 518 requirements in pyproject.toml yourself',
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

BASE_DIR = pathlib.Path(__file__).resolve().parent
DESCRIPTION = 'A fast TSP solver with Python bindings'


def parse_version() -> str:
    """Returns the version specified in CMakeLists.txt"""
    pattern = re.compile(r'project\(fast_tsp VERSION "([\d\.]+)"\)')
    cmake_file = BASE_DIR / 'CMakeLists.txt'
    lines = cmake_file.read_text(encoding='utf-8').splitlines()
    for line in lines:
        line_match = pattern.match(line)
        if line_match:
            return line_match.group(1)
    raise RuntimeError('Could not parse version')


# Import the README and use it as the long-description.
try:
    readme = BASE_DIR / 'README.md'
    long_description = readme.read_text(encoding='utf-8')
except FileNotFoundError:
    long_description = DESCRIPTION

VERSION = parse_version()

setup(
    name='fast_tsp',
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Soeren Mulvad',
    author_email='shmulvad@gmail.com',
    url='http://github.com/shmulvad/fast-tsp',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    cmake_install_dir='src/fast_tsp',
    package_data={'src': ['py.typed']},
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=['numpy>=1.0.0'],
    extras_require={
        'test': ['pytest', 'numpy'],
        'docs': ['sphinx', 'sphinx-rtd-theme', 'sphinx-autodoc-typehints'],
    },
)
