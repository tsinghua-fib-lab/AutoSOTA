PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
UV ?= uv
ifneq (,$(wildcard .env))
  include .env
  export
endif
# Auto-detect CUDA from nvcc if CUDA_HOME is not set; fall back to /usr/local/cuda.
NVCC_PATH := $(shell command -v nvcc 2>/dev/null)
CUDA_HOME ?= $(if $(NVCC_PATH),$(patsubst %/bin/nvcc,%,$(NVCC_PATH)),/usr/local/cuda)
CXX ?= g++
NVCC := $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= 89
CUDA_ARCHS ?= $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -E '^[0-9]+([.][0-9]+)?$$' | tr -d '. ' | sort -u)
CUDA_ARCHS := $(strip $(CUDA_ARCHS))
ifeq ($(CUDA_ARCHS),)
  CUDA_ARCHS := $(CUDA_ARCH)
endif
PTXAS_INFO ?= 1
REPO_ROOT := $(CURDIR)
BUILD_DIR := file_storage/build
MKDIR_P := mkdir -p

EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
TORCH_ABI := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TORCH_INC := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; import sysconfig; paths = include_paths(); py_inc = sysconfig.get_paths().get('include'); paths.append(py_inc) if py_inc else None; print(' '.join(f'-I{p}' for p in paths))")
TORCH_LIB_DIRS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join(f'-L{p}' for p in library_paths()))")
TORCH_RPATH := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join(f'-Wl,-rpath,{p}' for p in library_paths()))")
CUTLASS_INC := -I$(REPO_ROOT)/third_party/cutlass/include

GENCODE_FLAGS := $(foreach arch,$(CUDA_ARCHS),-gencode arch=compute_$(arch),code=compute_$(arch) -gencode arch=compute_$(arch),code=sm_$(arch))
PTXAS_FLAGS := $(if $(filter 1,$(PTXAS_INFO)),-Xptxas -v,)
CXXFLAGS := -O3 -std=c++17 -fPIC $(TORCH_INC) $(CUTLASS_INC) -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
NVCCFLAGS := -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr $(GENCODE_FLAGS) $(PTXAS_FLAGS) -Xcompiler -fPIC $(TORCH_INC) $(CUTLASS_INC) -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
LDFLAGS := $(TORCH_LIB_DIRS) $(TORCH_RPATH) -L$(CUDA_HOME)/lib64 -lcudart -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python
FWHT_REPO := https://github.com/Dao-AILab/fast-hadamard-transform.git
FWHT_REV := f134af63deb2df17e1171a9ec1ea4a7d8604d5ca
FWHT_PKG := fast-hadamard-transform @ git+$(FWHT_REPO)@$(FWHT_REV)

.PHONY: setup test test.small test.full kernels.build kernels.build.flashsketch kernels.build.flashblockrow \
	kernels.build.grass_sjlt kernels.clean bench.camera-ready.gram bench.camera-ready.ridge-regression \
	bench.camera-ready bench.ablation.gram bench.ablation.sketch-and-solve bench.ablation.ridge-regression \
	bench.ablation.ose bench.ablation bench.grass.external.mlp_mnist fig.camera-ready.gram \
	fig.camera-ready.ridge-regression fig.camera-ready.sketch-matrix fig.camera-ready.main \
	fig.camera-ready fig.ablation.gram.nolegend fig.ablation.sketch-and-solve.nolegend \
	fig.ablation.ridge-regression.nolegend fig.ablation.ose.nolegend fig.ablation.nolegend \
	fig.ablation.legend fig.ablation.summary-table fig.grass.camera_ready paper paper.clean

setup:
	$(UV) venv .venv
	$(UV) pip install --python .venv/bin/python setuptools packaging
	$(UV) pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.0+cu128 torchvision==0.24.0+cu128
	$(UV) pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu128 --no-build-isolation -r requirements.txt



test.small:
	$(PYTHON) -m pytest -m small

test.full:
	$(PYTHON) -m pytest -m full

test: test.small

kernels.build: kernels.build.flashsketch kernels.build.flashblockrow kernels.build.grass_sjlt


FLASHSKETCH_NAME := flashsketch_ext
FLASHSKETCH_DIR := $(BUILD_DIR)/$(FLASHSKETCH_NAME)
FLASHSKETCH_CPP := kernels/flashsketch/flashsketch_ext.cpp
FLASHSKETCH_CU := kernels/flashsketch/flashsketch_kernel.cu
FLASHSKETCH_CPP_OBJ := $(FLASHSKETCH_DIR)/flashsketch_ext.o
FLASHSKETCH_CU_OBJ := $(FLASHSKETCH_DIR)/flashsketch_kernel.cuda.o
FLASHSKETCH_SO := $(FLASHSKETCH_DIR)/$(FLASHSKETCH_NAME)$(EXT_SUFFIX)


FLASHBLOCKROW_NAME := flashblockrow_ext
FLASHBLOCKROW_DIR := $(BUILD_DIR)/$(FLASHBLOCKROW_NAME)
FLASHBLOCKROW_CPP := kernels/flashblockrow/flashblockrow_ext.cpp
FLASHBLOCKROW_CU := kernels/flashblockrow/flashblockrow_kernel.cu
FLASHBLOCKROW_CPP_OBJ := $(FLASHBLOCKROW_DIR)/flashblockrow_ext.o
FLASHBLOCKROW_CU_OBJ := $(FLASHBLOCKROW_DIR)/flashblockrow_kernel.cuda.o
FLASHBLOCKROW_SO := $(FLASHBLOCKROW_DIR)/$(FLASHBLOCKROW_NAME)$(EXT_SUFFIX)

GRASS_SJLT_NAME := grass_sjlt_ext
GRASS_SJLT_DIR := $(BUILD_DIR)/$(GRASS_SJLT_NAME)
GRASS_SJLT_CU := kernels/grass_sjlt/grass_sjlt_kernel.cu
GRASS_SJLT_CU_OBJ := $(GRASS_SJLT_DIR)/grass_sjlt_kernel.cuda.o
GRASS_SJLT_SO := $(GRASS_SJLT_DIR)/$(GRASS_SJLT_NAME)$(EXT_SUFFIX)

$(FLASHSKETCH_DIR) $(FLASHBLOCKROW_DIR) $(GRASS_SJLT_DIR):
	$(MKDIR_P) $@

$(FLASHSKETCH_CPP_OBJ): $(FLASHSKETCH_CPP) | $(FLASHSKETCH_DIR)
	$(CXX) $(CXXFLAGS) -DTORCH_EXTENSION_NAME=$(FLASHSKETCH_NAME) -c $< -o $@

$(FLASHSKETCH_CU_OBJ): $(FLASHSKETCH_CU) | $(FLASHSKETCH_DIR)
	$(NVCC) $(NVCCFLAGS) -DTORCH_EXTENSION_NAME=$(FLASHSKETCH_NAME) -c $< -o $@

$(FLASHSKETCH_SO): $(FLASHSKETCH_CPP_OBJ) $(FLASHSKETCH_CU_OBJ) | $(FLASHSKETCH_DIR)
	$(CXX) $^ -shared $(LDFLAGS) -o $@

$(FLASHBLOCKROW_CPP_OBJ): $(FLASHBLOCKROW_CPP) | $(FLASHBLOCKROW_DIR)
	$(CXX) $(CXXFLAGS) -DTORCH_EXTENSION_NAME=$(FLASHBLOCKROW_NAME) -c $< -o $@

$(FLASHBLOCKROW_CU_OBJ): $(FLASHBLOCKROW_CU) | $(FLASHBLOCKROW_DIR)
	$(NVCC) $(NVCCFLAGS) -DTORCH_EXTENSION_NAME=$(FLASHBLOCKROW_NAME) -c $< -o $@

$(FLASHBLOCKROW_SO): $(FLASHBLOCKROW_CPP_OBJ) $(FLASHBLOCKROW_CU_OBJ) | $(FLASHBLOCKROW_DIR)
	$(CXX) $^ -shared $(LDFLAGS) -o $@

$(GRASS_SJLT_CU_OBJ): $(GRASS_SJLT_CU) | $(GRASS_SJLT_DIR)
	$(NVCC) $(NVCCFLAGS) -DTORCH_EXTENSION_NAME=$(GRASS_SJLT_NAME) -c $< -o $@

$(GRASS_SJLT_SO): $(GRASS_SJLT_CU_OBJ) | $(GRASS_SJLT_DIR)
	$(CXX) $^ -shared $(LDFLAGS) -o $@

kernels.build.flashsketch: $(FLASHSKETCH_SO)

kernels.build.flashblockrow: $(FLASHBLOCKROW_SO)

kernels.build.grass_sjlt: $(GRASS_SJLT_SO)

kernels.clean:
	rm -rf file_storage/build

bench.camera-ready.gram: kernels.build
	$(PYTHON) bench/camera_ready/run_gram.py

bench.camera-ready.ridge-regression: kernels.build
	$(PYTHON) bench/camera_ready/run_ridge_regression.py

bench.camera-ready: bench.camera-ready.gram bench.camera-ready.ridge-regression

bench.ablation.gram: kernels.build
	$(PYTHON) bench/ablation/run_gram.py

bench.ablation.sketch-and-solve: kernels.build
	$(PYTHON) bench/ablation/run_sketch_solve.py

bench.ablation.ridge-regression: kernels.build
	$(PYTHON) bench/ablation/run_ridge_regression.py

bench.ablation.ose: kernels.build
	$(PYTHON) bench/ablation/run_ose_error.py

bench.ablation: bench.ablation.gram bench.ablation.sketch-and-solve bench.ablation.ridge-regression bench.ablation.ose

bench.grass.external.mlp_mnist: kernels.build
	$(PYTHON) bench/grass_external/run_mlp_mnist.py

bench.grass.external: bench.grass.external.mlp_mnist

fig.camera-ready.gram:
	$(PYTHON) analysis/figures_src/fig_e2e_camera_ready/make_figure.py

fig.camera-ready.ridge-regression:
	$(PYTHON) analysis/figures_src/fig_e2e_camera_ready_ridge_regression/make_figure.py

fig.camera-ready.sketch-matrix:
	$(PYTHON) analysis/figures_src/camera_ready/make_sketch_matrix_flashsketch.py

fig.camera-ready.main: fig.camera-ready.gram fig.camera-ready.ridge-regression fig.camera-ready.sketch-matrix

fig.camera-ready: fig.camera-ready.main

fig.ablation.gram.nolegend:
	$(PYTHON) analysis/figures_src/fig_ablation_gram/make_figure_nolegend.py

fig.ablation.sketch-and-solve.nolegend:
	$(PYTHON) analysis/figures_src/fig_ablation_sketch_solve/make_figure_nolegend.py


fig.ablation.ridge-regression.nolegend:
	$(PYTHON) analysis/figures_src/fig_ablation_ridge_regression/make_figure_nolegend.py

fig.ablation.ose.nolegend:
	$(PYTHON) analysis/figures_src/fig_ablation_ose/make_figure_nolegend.py

fig.ablation.nolegend: fig.ablation.gram.nolegend fig.ablation.sketch-and-solve.nolegend fig.ablation.ridge-regression.nolegend fig.ablation.ose.nolegend

fig.ablation.legend:
	$(PYTHON) analysis/figures_src/fig_ablation_legend/make_figure.py

fig.ablation.summary-table:
	$(PYTHON) analysis/figures_src/fig_ablation_summary_table/make_table.py

fig.grass.camera_ready:
	$(PYTHON) analysis/figures_src/fig_grass_camera_ready/make_figure.py

paper:
	mkdir -p paper/build
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory build main.tex
	cd paper && if grep -q "\\\\citation" build/main.aux; then bibtex build/main; else echo "No citations; skipping bibtex."; fi
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory build main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory build main.tex

paper.abstract:
	$(PYTHON) export_abstract_md.py

paper.clean:
	rm -rf paper/build
