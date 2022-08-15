#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp

# export VECLIB_MAXIMUM_THREADS=1

make -j H2_main
make -j H2_dtd

# for N in 8192; do
#     for matrix_type in 1; do
#         for admis in 0.4; do
#             for rank in 20; do
#                 lldb -o run -- ./bin/UMV_H2_far_dense $N $rank 128 $admis 3 geometry_admis 0 $matrix_type
#             done
#         done
#     done
# done

# for N in 8192; do
#     for matrix_type in 1; do
#         for admis in 1.2; do
#             for rank in 20; do
#                 lldb -o run -- ./bin/UMV_H2_far_dense $N $rank 128 $admis 3 geometry_admis 0 $matrix_type
#             done
#         done
#     done
# done
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
    nleaf=32
    ndim=2
    max_rank=30
    for N in 4096; do
        ./bin/H2_main --N $N \
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
        for nproc in 1 2 4 16 64; do
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
