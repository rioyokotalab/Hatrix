#!/bin/bash

make -j all

./bin/UMV_H2_far_dense 800 10 100 0.7 1 geometry_admis 0
# ./bin/UMV_H2_far_dense 800 10 100 0.7 1 geometry_admis 0

./bin/UMV_strong_H2_Nlevel 800 10 3 1
# ./bin/UMV_H2_far_dense 1600 10 100 0.7 1 geometry_admis 0
# ./bin/UMV_H2_far_dense 3200 10 100 0.7 1 geometry_admis 0

# ./bin/UMV_H2_far_dense 800 10 100 2 1 diagonal_admis 0
# ./bin/UMV_H2_far_dense 1600 10 100 2 1 diagonal_admis 0
# ./bin/UMV_H2_far_dense 3200 10 100 2 1 diagonal_admis 0
