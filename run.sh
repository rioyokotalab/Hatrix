#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J H2_UMV_FAR
#SBATCH --time=72:00:00

source ~/.bashrc

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_BUILD_TYPE=Release
make -j all

echo "GEOMETRY ADMIS"

for N in 500 1000 2000 4000 8000 12000 16000 32000; do
    for matrix_type in 1 0; do
        echo "CIRCLE GEOMETRY"
        for rank in 20 34 40; do
            ./bin/UMV_H2_far_dense $N $rank 100 1.2 3 geometry_admis 0 $matrix_type
        done

        echo "SPHERE GEOMETRY"
        for rank in 20 34 40; do
            ./bin/UMV_H2_far_dense $N $rank 100 1.2 3 geometry_admis 0 $matrix_type
        done

        echo "STARSH GRID DIAGO$NAL H2 DIM=2"
        for rank in 10 20 24 30 80; do
            ./bin/UMV_H2_far_dense $N $rank 100 2 2 diagonal_admis 1 $matrix_type
        done


        echo "STARSH GRID GEOMETRY DIM=2"
        for rank in 20 24 30 80; do
            ./bin/UMV_H2_far_dense $N $rank 100 0.7 2 geometry_admis 1 $matrix_type
        done

        echo "STARSH GRID GEOMETRY DIM=3"
        for rank in 40 50 60 80; do
            ./bin/UMV_H2_far_dense $N $rank 100 1 3 geometry_admis 1 $matrix_type
        done

        echo "STARSH SIN KERNEL DIM=2"
        for rank in 20 24 30 50 80; do
            ./bin/UMV_H2_far_dense $N $rank 100 1 2 diagonal_admis 2 $matrix_type
            ./bin/UMV_H2_far_dense $N $rank 100 2 2 diagonal_admis 2 $matrix_type
            ./bin/UMV_H2_far_dense $N $rank 100 3 2 diagonal_admis 2 $matrix_type
            ./bin/UMV_H2_far_dense $N $rank 100 1.2 2 geometry_admis 2 $matrix_type
            ./bin/UMV_H2_far_dense $N $rank 100 0.4 2 geometry_admis 2 $matrix_type
        done
    done
done
