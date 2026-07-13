#!/bin/bash

# Define parameters
D=2
L=1

echo "Running with parameters: n=5 to 20, D=$D, l=$L"

echo "Running generate_graph.py..."
python3 generate_graph.py  --degree $D 

echo "Running generate_test_samples.py..."
python3 generate_test_samples.py --D $D --l $L

echo "Running generate_test_cases.py..."
python3 generate_test_cases.py --D $D --l $L

echo "Running train_flooding.py..."
python3 train_flooding.py --D $D --l $L


echo "Done!"