#!/bin/bash

make -j all

rm result.txt

echo "------ UMV H2 NLEVEL ------"

# ./bin/UMV_H2_far_dense 512 3 64 0.8 2 geometry_admis 1 0
# ./bin/UMV_H2_far_dense 512 5 64 0.8 2 geometry_admis 1 0
# ./bin/UMV_H2_far_dense 512 10 64 0.8 2 geometry_admis 1 0
# ./bin/UMV_H2_far_dense 512 15 64 0.8 2 geometry_admis 1 0


# ./bin/UMV_H2_far_dense 1024 3 64 0.8 2 geometry_admis 1 0
# ./bin/UMV_H2_far_dense 1024 5 64 0.8 2 geometry_admis 1 0
# ./bin/UMV_H2_far_dense 1024 10 64 0.8 2 geometry_admis 1 0
# ./bin/UMV_H2_far_dense 1024 15 64 0.8 2 geometry_admis 1 0
# ./bin/UMV_H2_far_dense 1024 60 64 0.8 2 geometry_admis 1 0

# ./bin/UMV_H2_far_dense 512 10 64 0.5 2 geometry_admis 0 0
# ./bin/UMV_H2_far_dense 512 10 64 0.5 2 geometry_admis 0 1
./bin/UMV_H2_far_dense 512 10 64 1 2 geometry_admis 0 1
./bin/UMV_H2_far_dense 512 10 64 1 2 geometry_admis 0 0


# echo "SPHERE GEOMETRY"
# for rank in 34 40; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 1 3 geometry_admis 0 0
# done



# for rank in 34 35 36 37 38 39 40; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 1 3 geometry_admis 0 0
# done

# for rank in 14 15 16 20; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 1 2 geometry_admis 0 0
# done

# ./bin/UMV_H2_far_dense 256 12 32 1.3 3 geometry_admis 0 0
# for rank in 12 15 18 22 25; d
#     ./bin/UMV_H2_far_dense 256 $rank 32 1.3 3 geometry_admis 0 0
# done

# echo "STARSH GRID GEOMETRY DIM=1"
# for rank in 6 7 8 10 14; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 0.7 1 geometry_admis 1 0
# done

# echo "STARSH GRID DIAGONAL H2 DIM=2"
# for rank in 20 24 30; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 2 2 diagonal_admis 1 1
# done


# echo "STARSH GRID GEOMETRY DIM=2"
# for rank in 20 24 30; do
#     ./bin/UMV_H2_far_dense 1024 $rank 64 0.7 2 geometry_admis 1 1
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

cat result.txt
