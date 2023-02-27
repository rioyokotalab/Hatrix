#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp

# export VECLIB_MAXIMUM_THREADS=1

ulimit -c unlimited

make -j H2_dtd
make -j H2_main

for adm in 1.2; do
    nleaf=256
    ndim=3
    max_rank=100

    for N in 4096; do
        mpirun -n 1 ./bin/H2_dtd --N $N \
                      --nleaf $nleaf \
                      --kernel_func laplace \
                      --kind_of_geometry grid \
                      --ndim $ndim \
                      --max_rank $max_rank \
                      --accuracy -1 \
                      --admis $adm \
                      --admis_kind geometry \
                      --construct_algorithm miro \
                      --add_diag 1e-9  \
                      --kind_of_recompression 3 \
                      --use_nested_basis

        # ./bin/H2_main --N $N \
        #              --nleaf $nleaf \
        #              --kernel_func laplace \
        #              --kind_of_geometry grid \
        #              --ndim $ndim \
        #              --max_rank $max_rank \
        #              --accuracy -1 \
        #              --admis $adm \
        #              --admis_kind geometry \
        #              --construct_algorithm miro \
        #              --add_diag 1e-9  \
        #              --kind_of_recompression 3 \
        #              --use_nested_basis
    done
done
