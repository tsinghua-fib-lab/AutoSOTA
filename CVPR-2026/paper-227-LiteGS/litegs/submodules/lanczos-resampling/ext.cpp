/*
Copyright (c) 2025 Youyu Chen
SPDX-License-Identifier: MIT
*/
#include <torch/extension.h>
#include "cuda_lanczos/lanczos.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("fast_lanczos", &LanczosResampling);
}