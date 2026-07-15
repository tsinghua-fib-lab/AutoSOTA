#!/bin/bash
# Setup script for cuRegOT container reusability
# Ensures the conda cuDSS library is used instead of pip's incompatible version

set -e

CONDA_ENV=/opt/conda/envs/curegot
PIP_CUDSS_DIR=$CONDA_ENV/lib/python3.12/site-packages/nvidia/cu13/lib

echo "cuRegOT container setup"
echo "======================="

# Fix cuDSS library conflict: remove pip's cuDSS 0.8.0 to use conda's 0.7.1
if [ -f "$PIP_CUDSS_DIR/libcudss.so.0" ]; then
    echo "Removing pip-installed cuDSS libs (version 0.8.0, incompatible)..."
    rm -f "$PIP_CUDSS_DIR/libcudss.so.0"
    rm -f "$PIP_CUDSS_DIR/libcudss_commlayer_nccl.so.0"
    rm -f "$PIP_CUDSS_DIR/libcudss_commlayer_openmpi.so.0"
    rm -f "$PIP_CUDSS_DIR/libcudss_mtlayer_gomp.so.0"
    echo "Done. Conda cuDSS 0.7.1 will be used via LD_LIBRARY_PATH."
else
    echo "Pip cuDSS libs already removed."
fi

echo "CUDA_HOME=$CONDA_ENV"
echo "LD_LIBRARY_PATH includes $CONDA_ENV/lib"
echo ""
echo "To run evaluation:"
echo "  export CUDA_HOME=$CONDA_ENV"
echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib:\$LD_LIBRARY_PATH"
echo "  cd /repo && conda run -n curegot python eval_repro.py"
