#!/bin/bash

make -j all

# ./bin/HSS_Nlevel_construct 800 20 3
# ./bin/HSS_Nlevel_construct 1600 20 4
# ./bin/HSS_Nlevel_construct 3200 20 5

./bin/UMV_H2_far_dense 512 50 64 1 2 diagonal_admis 1
./bin/UMV_H2_far_dense 512 50 64 2 1 diagonal_admis 1
# ./bin/UMV_H2_far_dense 512 50 64 1 1 diagonal_admis 1
# ./bin/UMV_H2_far_dense 512 50 64 1 2 diagonal_admis 1

# ./bin/UMV_H2_far_dense 512 50 64 2 1 diagonal_admis 1
# ./bin/UMV_H2_far_dense 512 50 64 2 2 diagonal_admis 1

# ./bin/UMV_H2_far_dense 512 50 64 1 2 geometry_admis 1

# ./bin/UMV_strong_H2_Nlevel 512 50 3 2
