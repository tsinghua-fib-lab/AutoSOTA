# Copyright (c) 2025 Youyu Chen
# SPDX-License-Identifier: MIT
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
	name="FastLanczos", 
	packages=["FastLanczos"], 
	ext_modules=[
		CUDAExtension(
			name="FastLanczos._C", 
			sources=[
				"cuda_lanczos/lanczos.cu", 
				"ext.cpp"
			]
		)
	], 
	cmdclass={
		"build_ext": BuildExtension
	}
)