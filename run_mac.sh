#!/bin/bash

make -j HSS_main
export TMPDIR=/tmp

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

for nprocs in 1; do
    ./bin/HSS_main --N 128 \
                   --nleaf 32 \
                   --kernel_func laplace \
                   --kind_of_geometry col_file_3d \
                   --geometry_file /Users/sameer/57114x1.dat
                   --ndim 1 \
                   --max_rank 32 \
                   --accuracy 1e-11 \
                   --admis 0 \
                   --admis_kind diagonal \
                   --construct_algorithm miro \
                   --add_diag 1e-5 \
                   --use_nested_basis
done
