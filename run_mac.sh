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
cd build

cmake .. \
      -DCMAKE_EXE_LINKER_FLAGS=" -L/opt/homebrew/opt/libomp/lib" \
      -DGSL_INCLUDE_DIR="/Users/sameerdeshmukh/gitrepos/gsl-2.7.1/build/include" \
      -DGSL_LIBRARY="/Users/sameerdeshmukh/gitrepos/gsl-2.7.1/build/lib/libgsl.27.dylib" \
      -DLAPACKE_INCLUDE_DIR="/opt/homebrew/opt/lapack/include" \
      -DLAPACKE_LIBRARIES="/opt/homebrew/opt/lapack/lib"

make VERBOSE=1

# cmake .. \
#       -DCMAKE_EXE_LINKER_FLAGS=" -L/opt/homebrew/opt/libomp/lib -lomp " \
#       -DOpenMP_C_FLAGS="-fopenmp=lomp" \
#       -DOpenMP_CXX_FLAGS="-fopenmp=lomp" \
#       -DOpenMP_C_LIB_NAMES="libomp" \
#       -DOpenMP_CXX_LIB_NAMES="libomp" \
#       -DOpenMP_libomp_LIBRARY="/opt/local/lib/libomp.dylib" \
#       -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp  -I/opt/homebrew/opt/libomp/include" \
#       -DGSL_INCLUDE_DIR="/Users/sameerdeshmukh/gitrepos/gsl-2.7.1/build/include" \
#       -DGSL_LIBRARY="/Users/sameerdeshmukh/gitrepos/gsl-2.7.1/build/lib" \
#       -DLAPACKE_INCLUDE_DIR="/opt/homebrew/opt/lapack/include" \
#       -DLAPACKE_LIBRARIES="/opt/homebrew/opt/lapack/lib"


# export OMP_NUM_THREADS=16

# ROOT=$PWD
# cd examples/distributed/H2_construct
# export MPICC=mpicc

# $PARSEC_PATH/bin/parsec-ptgpp -E -i h2_factorize_flows.jdf -o h2_factorize_flows
# $MPICC $(pkg-config --cflags parsec) -I../include/distributed -O0 -g \
#        h2_factorize_flows.c -c -o h2_factorize_flows.o
# cd $ROOT

# make -j H2_construct

# for N in 2048; do
#     for MAX_RANK in 30; do
#         NLEAF=128
#         NDIM=1
#         KERNEL_FUNC=laplace
#         ADMIS_VALUE=0.3

#         # Laplace kernel parameters
#         p1=1e-3
#         mpirun -n 2 ./bin/H2_construct --N $N \
#                --ndim $NDIM \
#                --nleaf $NLEAF \
#                --max_rank $MAX_RANK \
#                --kernel_func $KERNEL_FUNC \
#                --kind_of_geometry grid \
#                --admis_kind geometry \
#                --admis $ADMIS_VALUE \
#                --geometry_file C60_fcc.xyz \
#                --param_1 $p1 \
#                --use_nested_basis 1
#     done
# done
