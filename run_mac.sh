#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig:/Users/sameer/Downloads/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib
export PATH=/Users/sameer/gitrepos/parsec/build/bin:$PATH

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp
ROOT=$PWD

set -e

make clean

# export VECLIB_MAXIMUM_THREADS=1
cd examples/distributed/H2_ptg
./compile_jdf.sh
cd $ROOT

ulimit -c unlimited

make -j H2_main
# make -j H2_dtd

nleaf=256
max_rank=50
ndim=3
adm=1

for N in 1024; do
    ./bin/H2_main --N $N \
                 --nleaf $nleaf \
                 --kernel_func laplace \
                 --kind_of_geometry grid \
                 --ndim $ndim \
                 --max_rank $max_rank \
                 --accuracy -1 \
                 --admis $adm \
                 --admis_kind diagonal \
                 --construct_algorithm miro \
                 --param_1 1e-9  \
                 --kind_of_recompression 3


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
    #              --add_diag 1e-9  \
    #              --kind_of_recompression 3
done
