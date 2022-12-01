#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp

# export VECLIB_MAXIMUM_THREADS=1

make -j H2_main
make -j H2_dtd

for adm in 0.8; do
    nleaf=512
    ndim=3
    max_rank=110

    for N in 8192; do
        ./bin/H2_main --N $N \
                      --nleaf $nleaf \
                      --kernel_func laplace \
                      --kind_of_geometry grid \
                      --ndim $ndim \
                      --max_rank $max_rank \
                      --accuracy 1e-8 \
                      --admis $adm \
                      --admis_kind geometry \
                      --construct_algorithm miro \
                      --add_diag 1e-10 \
                      --use_nested_basis
    done
done
