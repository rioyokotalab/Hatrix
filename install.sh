#!/bin/bash

VAR_FILE=hatrix_deps.sh

rm -rf dependencies
mkdir -p dependencies
cd dependencies

# Build googletest
git clone https://github.com/google/googletest.git -b release-1.10.0
cd googletest
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
make install
cd ..                           # out of build
cd ..                           # out of googletest
cd ..                           # root folder

echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:$PWD/dependencies/googletest/build/lib/pkgconfig" > $VAR_FILE

source $VAR_FILE
make all
./bin/gemm
