#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig:/Users/sameer/gitrepos/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib
export PATH=/Users/sameer/gitrepos/parsec/build/bin:$PATH

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp
ROOT=$PWD

set -e

make clean

# export VECLIB_MAXIMUM_THREADS=1
# cd examples/distributed/H2_ptg
# ./compile_jdf.sh
# cd $ROOT

ulimit -c unlimited

make -j H2_main
# make -j H2_dtd

max_rank=50
ndim=2
adm=2

for N in 32768; do
    for nleaf in 512 1024 2048; do
        for max_rank in 50 100 150 200; do
            ./bin/H2_main --N $N \
                          --nleaf $nleaf \
                          --kernel_func gsl_matern \
                          --kind_of_geometry grid \
                          --ndim $ndim \
                          --max_rank $max_rank \
                          --accuracy -1 \
                          --admis $adm \
                          --admis_kind geometry \
                          --construct_algorithm miro \
                          --param_1 1e-2 --param_2 0.5 --param_3 0.1 \
                          --kind_of_recompression 3 \
                          --use_nested_basis
        done
    done
    # ./bin/H2_ptg --N $N \
    #              --nleaf $nleaf \
    #              --kernel_func laplace \
    #              --kind_of_geometry grid \
    #              --ndim $ndim \
    #              --max_rank $max_rank \
    #              --accuracy -1 \
    #              --admis $adm \
    #              --admis_kind diagonal \
    #              --construct_algorithm miro \
    #              --param_1 1e-9  \
    #              --kind_of_recompression 3
done
