#!/bin/bash
echo "Creating conda environment..."
conda env create -f environment_basic.yaml
ENV_NAME=$(grep "name:" environment_basic.yaml | cut -d' ' -f2)
conda run -n "$ENV_NAME" pip install pandas==2.2.2