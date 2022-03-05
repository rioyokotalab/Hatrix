#!/bin/bash

make -j all

rm result.txt

echo "------ UMV H2 NLEVEL ------"

echo "CIRCLE GEOMETRY"
for rank in 20 34 40; do
    ./bin/UMV_H2_far_dense 1024 $rank 64 1.2 3 geometry_admis 0 1
done

echo "SPHERE GEOMETRY"
for rank in 34 40; do
    ./bin/UMV_H2_far_dense 1024 $rank 64 1.2 3 geometry_admis 0 1
done

echo "STARSH GRID DIAGONAL H2 DIM=2"
for rank in 20 24 30; do
    ./bin/UMV_H2_far_dense 1024 $rank 64 2 2 diagonal_admis 1 1
done


echo "STARSH GRID GEOMETRY DIM=2"
for rank in 20 24 30; do
    ./bin/UMV_H2_far_dense 1024 $rank 64 0.7 2 geometry_admis 1 1
done

echo "STARSH GRID GEOMETRY DIM=3"
for rank in 40 50 60; do
    ./bin/UMV_H2_far_dense 1024 $rank 64 1 3 geometry_admis 1 0
done

echo "STARSH SIN KERNEL DIM=2"
for rank in 20 24 30; do
    ./bin/UMV_H2_far_dense 1024 $rank 64 0.7 2 geometry_admis 2 1
done

cat result.txt
