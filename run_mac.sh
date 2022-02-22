#!/bin/bash

make -j all

# ./bin/HSS_Nlevel_construct 800 20 3
# ./bin/HSS_Nlevel_construct 1600 20 4
# ./bin/HSS_Nlevel_construct 3200 20 5

echo "---- SQ EXP NDIM=2 ADMIS=1 KERNEL FUNC=1 ----"
# ./bin/UMV_H2_far_dense 256 25 32 1 2 diagonal_admis 1 0
# ./bin/UMV_H2_far_dense 256 25 32 1 1 diagonal_admis 1 0

# ./bin/UMV_H2_far_dense 512 50 64 1 2 geometry_admis 0 0
# ./bin/UMV_H2_far_dense 512 20 64 1 2 geometry_admis 0 0
./bin/UMV_H2_far_dense 1024 30 64 1 3 geometry_admis 1 0

# ./bin/UMV_H2_far_dense 64 5 8 1 2 geometry_admis 0 0
