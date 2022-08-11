#!/bin/bash

make -j HSS_main
export TMPDIR=/tmp


make -j H2_main

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

for adm in 0.8; do
    ./bin/H2_main --N 1024 \
         --nleaf 128 \
         --kernel_func laplace \
         --kind_of_geometry circular \
         --ndim 2 \
         --max_rank 50 \
         --accuracy 1e-11 \
         --admis $adm \
         --admis_kind geometry \
         --construct_algorithm miro \
         --add_diag 1e-7 \
         --use_nested_basis
done

# profile strumpack and see where N^2 is happening.
