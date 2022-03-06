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

for N in 1024 2048 4096 8192 16384 32768; do
    for matrix_type in 1 0; do
        echo "CIRCLE GEOMETRY"
        for rank in 20 34 40; do
            ./examples/UMV_H2_far_dense $N $rank 128 1.2 3 geometry_admis 0 $matrix_type
        done

        echo "SPHERE GEOMETRY"
        for rank in 20 34 40; do
            ./examples/UMV_H2_far_dense $N $rank 128 1.2 3 geometry_admis 0 $matrix_type
        done

        for dim in 2 3; do
            echo "STARSH GRID DIAGO$NAL H2 DIM=$dim"
            for beta in 0.3 0.5 1; do
                for nu in 0.5 1; do
                    for sigma in 1.0 1.5; do
                        for rank in 10 20 24 30 80; do
                            ./examples/UMV_H2_far_dense $N $rank 128 2 $dim diagonal_admis 1 $matrix_type $beta $nu $sigma
                        done

                        for rank in 20 24 30 80; do
                            for admis in 0.7 1; do
                                ./examples/UMV_H2_far_dense $N $rank 128 $admis $dim geometry_admis 1 $matrix_type $beta $nu $sigma
                            done
                        done
                    done
                done
            done
        done


        echo "STARSH SIN KERNEL DIM=2"
        for rank in 20 24 30 50 80; do
            ./examples/UMV_H2_far_dense $N $rank 128 1 2 diagonal_admis 2 $matrix_type
            ./examples/UMV_H2_far_dense $N $rank 128 2 2 diagonal_admis 2 $matrix_type
            ./examples/UMV_H2_far_dense $N $rank 128 3 2 diagonal_admis 2 $matrix_type
            ./examples/UMV_H2_far_dense $N $rank 128 1.2 2 geometry_admis 2 $matrix_type
            ./examples/UMV_H2_far_dense $N $rank 128 0.4 2 geometry_admis 2 $matrix_type
        done
    done
done
