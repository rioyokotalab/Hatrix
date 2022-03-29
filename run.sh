#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J ATAT
#SBATCH --time=72:00:00

source ~/.bashrc

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_BUILD_TYPE=Release
make -j all

for N in 1024 2048 4096 8192 16384 32768 65536 131072; do
    for matrix_type in 1 0; do
        for admis in 0 1.2 1.5 2; do
            for rank in 5 10 20 40 70 80; do
                ./examples/UMV_H2_far_dense $N $rank 128 $admis 3 geometry_admis 0 $matrix_type
            done
        done
    done
done
