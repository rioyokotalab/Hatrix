#!/bin/bash

PARSEC_PATH=/Users/sameerdeshmukh/gitrepos/parsec/install

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$PARSEC_PATH/lib/pkgconfig:/Users/sameerdeshmukh/gitrepos/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PARSEC_PATH/lib
export PATH=$PARSEC_PATH/bin:$PATH

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp
ROOT=$PWD

# set -e

# rm -rf build
mkdir build

pushd build
cmake .. \
      -DCMAKE_EXE_LINKER_FLAGS=" -L/opt/homebrew/opt/libomp/lib" \
      -DCMAKE_CXX_FLAGS=" -I/opt/homebrew/opt/libomp/include " \
      -DGSL_INCLUDE_DIR="/Users/sameerdeshmukh/gitrepos/gsl-2.7.1/build/include" \
      -DGSL_LIBRARY="/Users/sameerdeshmukh/gitrepos/gsl-2.7.1/build/lib/libgsl.27.dylib" \
      -DLAPACKE_INCLUDE_DIR="/opt/homebrew/opt/lapack/include" \
      -DLAPACKE_LIBRARIES="/opt/homebrew/opt/lapack/lib" \
      -DCMAKE_BUILD_TYPE=Debug
make -j VERBOSE=1
popd

# N nleaf acc max_rank admis kernel geom ndim matrix_type

for N in 512 2048; do
    ./build/examples/H2_strong_CON_sameer \
        $N 64 0 40 0.5 0 1 3 1
done
