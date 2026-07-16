#!/bin/bash

tar -xvf run/static_data/encodings.tar.gz
tar -xvf run/static_data/selected.tar.gz
mv encodings/ run/static_data/
mv selected/ run/static_data/

git submodule init
git submodule update

export CUDA 

pip install --upgrade pip
pip install torch torchvision torchaudio xformers || exit 1
pip install --upgrade setuptools

echo "=> Installing cpp-hash..."
cd submodules/cpp-hash || exit 1
python setup.py install || exit 1
cd ../..

echo "=> Installing vllm..."
pip install submodules/vllm || exit 1


# Install specific versions of packages for VLLM compatibility
pip install lingua-language-detector tiktoken transformers scikit-learn nltk pyinflect accelerate openai textattack pandas dacite dahuffman argostranslate dill mauve-text faiss-gpu sacrebleu sacremoses psutil pyext

#pip install starlette==0.27.0

echo "=> Installing APPS..."
pip install -e apps/ || exit 1

echo "=> Installing watermark-benchmark..."
python setup.py install || exit 1

pip install --upgrade openai httpcore
