#!/bin/bash

make -j all

# ./bin/HSS_Nlevel_construct 800 20 3
# ./bin/HSS_Nlevel_construct 1600 20 4
# ./bin/HSS_Nlevel_construct 3200 20 5

echo "---- SQ EXP NDIM=2 ADMIS=1 KERNEL FUNC=1 ----"
# ./bin/UMV_H2_far_dense 256 25 32 1 2 diagonal_admis 1 0
# ./bin/UMV_H2_far_dense 256 25 32 1 1 diagonal_admis 1 0

./bin/UMV_H2_far_dense 512 50 64 1 2 diagonal_admis 1 1
./bin/UMV_H2_far_dense 512 50 64 2 2 diagonal_admis 1 1
# ./bin/UMV_H2_far_dense 512 50 64 2 2 diagonal_admis 1 1
# ./bin/UMV_H2_far_dense 512 50 64 1 1 diagonal_admis 1 0

# echo "---- SQ EXP NDIM=2 ADMIS=2 KERNEL FUNC=1 ----"
# ./bin/UMV_H2_far_dense 256 20 32 2 2 diagonal_admis 1 0
# ./bin/UMV_H2_far_dense 512 50 64 1 1 diagonal_admis 1
# ./bin/UMV_H2_far_dense 512 50 64 1 2 diagonal_admis 1
