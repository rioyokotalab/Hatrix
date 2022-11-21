#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp

export VECLIB_MAXIMUM_THREADS=1

make -j H2_main
make -j H2_dtd

for adm in 1.8; do
    nleaf=128
    ndim=1
    max_rank=100
    for N in 1024; do
        mpirun -n 1 ./bin/H2_dtd --N $N \
                      --nleaf $nleaf \
                      --kernel_func laplace \
                      --kind_of_geometry circular \
                      --ndim $ndim \
                      --max_rank $max_rank \
                      --accuracy 1e-13 \
                      --admis $adm \
                      --admis_kind geometry \
                      --construct_algorithm miro \
                      --add_diag 1e-8 \
                      --use_nested_basis
    done
done
