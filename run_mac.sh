#!/bin/bash

make -j all

# ./bin/UMV_H2_far_dense 1024 70 128 1 1 diagonal_admis 0
./bin/UMV_H2_far_dense 1024 50 128 0 1 diagonal_admis 1
# ./bin/UMV_H2_far_dense 1024 70 128 1 2 diagonal_admis 1
# ./bin/UMV_H2_far_dense 1024 120 128 1 3 diagonal_admis 1

./bin/UMV_H2_far_dense 1024 50 128 1 1 geometry_admis 1
# ./bin/UMV_H2_far_dense 1024 70 128 1 2 geometry_admis 1
# ./bin/UMV_H2_far_dense 1024 120 128 1 3 diagonal_admis 1
# ./bin/UMV_strong_H2_Nlevel 800 10 3 1
# ./bin/UMV_H2_far_dense 80 6 10 1 2 geometry_admis 1
# ./bin/UMV_strong_chained_product 100 4 10 0
# ./bin/UMV_H2_far_dense 1600 10 100 1 1 diagonal_admis 0
# ./bin/UMV_H2_far_dense 1600 10 100 0.7 1 geometry_admis 1
# ./bin/UMV_H2_far_dense 800 10 100 0.7 1 geometry_admis 0


# ./bin/UMV_strong_H2_Nlevel 1600 10 4 1
# ./bin/UMV_H2_far_dense 1600 10 100 0.7 1 geometry_admis 0
# ./bin/UMV_H2_far_dense 3200 10 100 0.7 1 geometry_admis 0

# ./bin/UMV_H2_far_dense 800 10 100 2 1 diagonal_admis 0
# ./bin/UMV_H2_far_dense 1600 10 100 2 1 diagonal_admis 0
# ./bin/UMV_H2_far_dense 3200 10 100 2 1 diagonal_admis 0
