#!/bin/bash

PARSEC_PATH=/Users/sameer/gitrepos/parsec/install

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$PARSEC_PATH/lib/pkgconfig:/Users/sameer/gitrepos/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PARSEC_PATH/lib
export PATH=$PARSEC_PATH/bin:$PATH

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp
ROOT=$PWD

set -e

# export OMP_NUM_THREADS=16

ROOT=$PWD
cd examples/distributed/H2_construct
export MPICC=mpicc

$PARSEC_PATH/bin/parsec-ptgpp -E -i h2_factorize_flows.jdf -o h2_factorize_flows
$MPICC $(pkg-config --cflags parsec) -I../include/distributed -O0 -g \
       h2_factorize_flows.c -c -o h2_factorize_flows.o
cd $ROOT

make -j H2_construct

for N in 1024; do
    for MAX_RANK in 30; do
        NLEAF=128
        NDIM=1
        KERNEL_FUNC=laplace
        ADMIS_VALUE=0.3

        # Laplace kernel paramters
        p1=1e-3
        mpirun -n 4 ./bin/H2_construct --N $N \
               --ndim $NDIM \
               --nleaf $NLEAF \
               --max_rank $MAX_RANK \
               --kernel_func $KERNEL_FUNC \
               --kind_of_geometry grid \
               --admis_kind geometry \
               --admis $ADMIS_VALUE \
               --geometry_file C60_fcc.xyz \
               --param_1 $p1 \
               --use_nested_basis 0
    done
done
