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
# echo "LINE GEOMETRY"
# for rank in 6 7 8 10 14; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 0.7 1 geometry_admis 1 0
# done


echo "2D GRID GEOMETRY"
for rank in 10; do
    ./bin/UMV_H2_far_dense 128 $rank 16 0.9 2 geometry_admis 1 0
done

# echo "SPHERE GEOMETRY"
# for rank in 34 35 36 37 38 39 40; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 1 3 geometry_admis 1 0
# done

# for rank in 34 35 36 37 38 39 40; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 1 3 geometry_admis 0 0
# done

# for rank in 14 15 16 20; do
#     ./bin/UMV_H2_far_dense 512 $rank 32 1 3 geometry_admis 0 0
# done

# ./bin/UMV_H2_far_dense 256 12 32 1.3 3 geometry_admis 0 0
# for rank in 12 15 18 22 25; do
#     ./bin/UMV_H2_far_dense 256 $rank 32 1.3 3 geometry_admis 0 0
# done

# echo "STARSH GRID GEOMETRY DIM=1"
# for rank in 6 7 8 10 14; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 0.7 1 geometry_admis 1 0
# done

# echo "STARSH GRID GEOMETRY DIM=2"
# for rank in 20 24 30; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 0.7 2 geometry_admis 1 0
# done

# echo "STARSH GRID GEOMETRY DIM=3"
# for rank in 40 50 60; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 1 3 geometry_admis 1 0
# done

# for rank in 40 50 60; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 0.8 3 geometry_admis 1 0
# done

# for rank in 40 50 60; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 1 3 diagonal_admis 1 0
# done
