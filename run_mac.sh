#!/bin/bash

make -j all

./bin/UMV_H2_far_dense 800 20 100 0.7 1 geometry_admis 0

# ./bin/H2_far_dense_construct 800 5 100 2 1 diagonal_admis
# ./bin/H2_far_dense_construct 1600 5 100 2 1 diagonal_admis
# ./bin/H2_far_dense_construct 3200 5 100 2 1 diagonal_admis

# ./bin/H2_far_dense_construct 800 20 50 0.7 2 geometry_admis 1
# ./bin/H2_far_dense_construct 1600 20 50 0.7 2 geometry_admis 1

# echo "CONSTANT N"

# for rank in 10 20 30 40; do
#     for admis in 0.4 0.5 0.7; do
#         ./bin/H2_far_dense_construct 3200 $rank 50 $admis 2 geometry_admis 1
#     done
# done


# for rank in 10 20 30 40; do
#     for admis in 0.4 0.5 0.7; do
#         ./bin/H2_far_dense_construct 3200 $rank 50 $admis 3 geometry_admis 1
#     done
# done

# echo "CONSTANT RANK"

# for rank in 30; do
#     for admis in 0.5; do
#         ./bin/H2_far_dense_construct 3200 $rank 50 $admis 3 geometry_admis 1
#         ./bin/H2_far_dense_construct 6400 $rank 50 $admis 3 geometry_admis 1
#         ./bin/H2_far_dense_construct 12800 $rank 50 $admis 3 geometry_admis 1
#         ./bin/H2_far_dense_construct 25600 $rank 50 $admis 3 geometry_admis 1
#     done
# done


# ./bin/H2_far_dense_construct 1600 5 100 0.5 1 geometry_admis
# ./bin/H2_far_dense_construct 3200 5 100 0.5 1 geometry_admis
# ./bin/H2_far_dense_construct 6400 5 100 0.5 1 geometry_admis
# ./bin/H2_far_dense_construct 1600 15 100 1 3 geometry_admis

# ./bin/H2_far_dense_construct 6400 15 100 1 3 geometry_admis
