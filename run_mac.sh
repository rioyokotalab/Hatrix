#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp

# export VECLIB_MAXIMUM_THREADS=1

make -j H2_main
make -j H2_dtd

    # ./bin/H2_main --N $N \
    #        --nleaf $nleaf \
    #        --kernel_func laplace \
    #        --kind_of_geometry circular \
    #        --ndim $ndim \
    #        --max_rank $max_rank \
    #        --accuracy 1e-11 \
    #        --admis $adm \
    #        --admis_kind geometry \
    #        --construct_algorithm miro \
    #        --add_diag 1e-7 \
    #        --use_nested_basis

for adm in 0.8; do
    nleaf=64
    ndim=2
    max_rank=48
    for N in 8192; do
        # ./bin/H2_main --N $N \
        #               --nleaf $nleaf \
        #               --kernel_func laplace \
        #               --kind_of_geometry circular \
        #               --ndim $ndim \
        #               --max_rank $max_rank \
        #               --accuracy 1e-11 \
        #               --admis $adm \
        #               --admis_kind geometry \
        #               --construct_algorithm miro \
        #               --add_diag 1e-9 \
        #               --use_nested_basis
        for nproc in 64; do
            mpirun -n $nproc --oversubscribe ./bin/H2_dtd --N $N \
                   --nleaf $nleaf \
                   --kernel_func laplace \
                   --kind_of_geometry circular \
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
done

# profile strumpack and see where N^2 is happening.
