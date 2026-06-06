#!/bin/bash

# Installation script for Circuit Stability Codebase
# This script sets up the environment for the research codebase

set -e  # Exit on any error

echo "🚀 Setting up Circuit Stability Codebase Environment"
echo "=================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Check if we're in the right directory (should contain environment.yml)
if [ ! -f "environment.yml" ]; then
    echo "❌ environment.yml not found. Please run this script from the repository root."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Check for Graphviz
if ! command -v dot &> /dev/null; then
    echo "⚠️  Graphviz not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install graphviz
        else
            echo "❌ Homebrew not found. Please install Graphviz manually:"
            echo "   brew install graphviz"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y graphviz
        elif command -v yum &> /dev/null; then
            sudo yum install -y graphviz
        else
            echo "❌ Package manager not found. Please install Graphviz manually."
            exit 1
        fi
    else
        echo "❌ Unsupported OS. Please install Graphviz manually."
        exit 1
    fi
else
    echo "✅ Graphviz already installed"
fi

# Create conda environment
echo "📦 Creating conda environment..."
conda env create -f environment.yml

echo "✅ Environment created successfully!"
echo ""
echo "🎉 Setup complete! To activate the environment, run:"
echo "   conda activate ml"
echo ""
echo "🔍 To verify the installation, run:"
echo "   python -c \"import torch; import transformer_lens; import pygraphviz; print('All dependencies installed successfully!')\""
echo ""
echo "📚 Next steps:"
echo "   1. Activate the environment: conda activate ml"
echo "   2. Check out the notebooks/ directory for examples"
echo "   3. Run experiments using scripts in src/experiments/" 