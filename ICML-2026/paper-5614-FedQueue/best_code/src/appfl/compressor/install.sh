#!/bin/bash

# Create a directory for all installation
cd ~
mkdir -p .appfl/.compressor
cd .appfl/.compressor

# Install ZFP
if pip show "zfpy" >/dev/null 2>&1; then
    echo -e "\033[32mzfpy is already installed.\033[0m"
else
    # If the package is not installed, install it
    echo -e "\033[32mInstalling zfpy...\033[0m"
    pip install zfpy
    echo -e "\033[32mzfpy installation done.\033[0m"
    echo "======================"
fi

# Install SZ2
if [ -d "SZ" ]; then
    echo -e "\033[32mSZ2 is already installed.\033[0m"
else
    echo -e "\033[32mInstalling SZ2...\033[0m"
    git clone https://github.com/szcompressor/SZ.git && cd SZ
    mkdir -p build && cd build
    cmake -DCMAKE_INSTALL_PREFIX:PATH=.. ..
    make
    make install
    echo -e "\033[32mSZ2 installation done.\033[0m"
    echo "======================"
    cd ../..
fi

# Install SZ3
if [ -d "SZ3" ]; then
    echo -e "\033[32mSZ3 is already installed.\033[0m"
else
    echo -e "\033[32mInstalling SZ3...\033[0m"
    git clone https://github.com/szcompressor/SZ3.git && cd SZ3
    mkdir -p build && cd build
    cmake -DCMAKE_INSTALL_PREFIX:PATH=.. ..
    make
    make install
    echo -e "\033[32mSZ3 installation done.\033[0m"
    echo "======================"
    cd ../..
fi

# Install SZx
echo -e "\033[31mSZx is not open source, please install it manually by contacting the author.\033[0m"
