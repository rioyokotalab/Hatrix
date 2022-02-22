#!/bin/bash

make -j all

# ./bin/HSS_Nlevel_construct 800 20 3
# ./bin/HSS_Nlevel_construct 1600 20 4
# ./bin/HSS_Nlevel_construct 3200 20 5

echo "---- SQ EXP NDIM=2 ADMIS=1 KERNEL FUNC=1 ----"

# ./bin/UMV_H2_far_dense 2000 25 50 0.4 2 geometry_admis 2 0
# ./bin/UMV_H2_far_dense 4000 25 50 0.4 2 geometry_admis 2 0
# ./bin/UMV_H2_far_dense 1920 20 30 1.6 3 geometry_admis 0 0


# ./bin/UMV_H2_far_dense 64 5 8 1 2 geometry_admis 0 0

# echo "CIRCLE GEOMETRY"
# for rank in 6 7 8 10 14; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 0.7 2 geometry_admis 0 0
# done

# echo "SPHERE GEOMETRY"
# for rank in 10 20 25 30 40 50 60; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 1 3 geometry_admis 0 0
# done

# echo "STARSH GRID GEOMETRY DIM=1"
# for rank in 6 7 8 10 14; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 0.7 1 geometry_admis 1 0
# done

# echo "STARSH GRID GEOMETRY DIM=2"
# for rank in 20 24 30; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 0.7 2 geometry_admis 1 0
# done

echo "STARSH GRID GEOMETRY DIM=3"
for rank in 20 24 30; do
    ./bin/UMV_H2_far_dense 1024 $rank 64 1 3 geometry_admis 1 0
done
