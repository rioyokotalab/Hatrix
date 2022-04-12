#!/bin/bash

make test
# make -j
# ./bin/HSS_main --N 1024 --nleaf 128 --kernel-func laplace --add-diag 1e-4 \
#                --rank 10 --use-nested-basis 1
# ./bin/HSS_main --N 1024 --nleaf 128 --kernel-func laplace --add-diag 1e-4 --rank 20


# ./bin/UMV_weak_Nlevel 320 15 4
# ./bin/UMV_weak_Nlevel 640 15 5
# ./bin/UMV_weak_Nlevel 1280 15 6

# rm result.txt

# ./bin/UMV_H2_far_dense 1024 10 128 0 1 diagonal_admis 0 1
# ./bin/UMV_H2_far_dense 1024 20 128 0 1 diagonal_admis 0 1

# echo "------ UMV H2 NLEVEL ------"
# for matrix_type in 1; do
#     echo "CIRCLE GEOMETRY"
#     for rank in 20 34 40; do
#         ./bin/UMV_H2_far_dense 2048 $rank 128 0 1 diagonal_admis 0 $matrix_type
#     done
# done
