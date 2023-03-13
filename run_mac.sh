#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib
export PATH=/Users/sameer/gitrepos/parsec/build/bin:$PATH

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp
ROOT=$PWD

set -e

# export VECLIB_MAXIMUM_THREADS=1
cd examples/distributed/H2_ptg
./compile_jdf.sh
cd $ROOT

ulimit -c unlimited

# make -j H2_ptg
make -j H2_dtd

nleaf=256
max_rank=50
ndim=3
adm=1

for N in 8192; do
    mpirun --oversubscribe -n 4 ./bin/H2_dtd --N $N \
                      --nleaf $nleaf \
                      --kernel_func laplace \
                      --kind_of_geometry grid \
                      --ndim $ndim \
                      --max_rank $max_rank \
                      --accuracy -1 \
                      --admis $adm \
                      --admis_kind diagonal \
                      --construct_algorithm miro \
                      --add_diag 1e-9  \
                      --kind_of_recompression 3 \
                      --use_nested_basis

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
