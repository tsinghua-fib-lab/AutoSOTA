#!/bin/bash

# Define parameters
N=7
D=2
L=1

echo "Running with parameters: n=$N, D=$D, l=$L"

echo "Running generate_test_samples.py..."
python3 generate_test_samples.py --n $N --D $D --l $L

echo "Running generate_test_cases.py..."
python3 generate_test_cases.py --n $N --D $D --l $L

echo "Running train_flooding.py..."
python3 train_flooding.py --n $N --D $D --l $L


echo "Done!"