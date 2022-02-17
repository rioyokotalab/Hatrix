#!/bin/bash

make -j all

# ./bin/UMV_H2_far_dense 512 50 64 1 2 diagonal_admis 1

./bin/UMV_H2_far_dense 512 30 64 1 2 diagonal_admis 1
./bin/UMV_H2_far_dense 512 30 64 1 2 geometry_admis 1
