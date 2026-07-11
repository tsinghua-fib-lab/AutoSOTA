#!/usr/bin/env python3

CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

DESCRIPTION = 'Generates knockoff with neural networks.'

NUMPY_MIN_VERSION  = '1.14.0'
TORCH_MIN_VERSION  = '0.4.1'
CVXPY_MIN_VERSION  = '1.0.6'
SCIPY_MIN_VERSION  = '1.0.0'

REQUIRES            = ["numpy (>=%s)" % NUMPY_MIN_VERSION,
                       "torch (>=%s)" % TORCH_MIN_VERSION,
                       "cvxpy (>=%s)" % CVXPY_MIN_VERSION,
                       "scipy (>=%s)" % SCIPY_MIN_VERSION]
                    
from setuptools import Extension
from setuptools import setup
import sys

try: 
    import torch
except:
    print("You don't seem to have torch installed. Please install it.")
    sys.exit(1)

# Speficy dependencies
DEPENDENCIES = ['numpy>='+NUMPY_MIN_VERSION, 'torch>='+TORCH_MIN_VERSION, 'cvxpy>='+CVXPY_MIN_VERSION,
                'scipy>='+SCIPY_MIN_VERSION]

################ COMPILE

def main(**extra_args):
    setup(name='DeepKnockoffs',
          maintainer="Matteo Sesia",
          maintainer_email="msesia@stanford.edu",
          description=DESCRIPTION,
          url="https://web.stanford.edu/group/candes/deep-knockoffs/",
          download_url="",
          license="GPL-v3 license",
          classifiers=CLASSIFIERS,
          author="Matteo Sesia, Yaniv Romano",
          author_email="msesia@stanford.edu, yromano@stanford.edu",
          platforms="OS Independent",
          version='0.1.0',
          requires=REQUIRES,
          provides=["DeepKnockoffs"],
          packages     = ['DeepKnockoffs',
                         ],
          package_data = {},
          data_files=[],
          scripts= [],
          long_description = open('README.rst', 'rt').read(),
          install_requires = DEPENDENCIES
    )

#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main()
