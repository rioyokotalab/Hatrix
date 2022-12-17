#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp

# export VECLIB_MAXIMUM_THREADS=1

ulimit -c unlimited

make -j H2_main
make -j H2_dtd

for adm in 0.7; do
    nleaf=256
    ndim=2
    max_rank=200

    for N in 8192; do
        # ./bin/H2_main --N $N \
        #               --nleaf $nleaf \
        #               --kernel_func laplace \
        #               --kind_of_geometry grid \
        #               --ndim $ndim \
        #               --max_rank $max_rank \
        #               --accuracy 1e-11 \
        #               --admis $adm \
        #               --admis_kind geometry \
        #               --construct_algorithm miro \
        #               --add_diag 1e-9 \
        #               --use_nested_basis


        ./bin/H2_dtd --N $N \
                      --nleaf $nleaf \
                      --kernel_func laplace \
                      --kind_of_geometry grid \
                      --ndim $ndim \
                      --max_rank $max_rank \
                      --accuracy 1e-11 \
                      --admis $adm \
                      --admis_kind geometry \
                      --construct_algorithm miro \
                      --add_diag 1e-9 \
                      --use_nested_basis
    done
done
