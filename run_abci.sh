#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -N HATRIX
#$ -o HATRIX_out.log
#$ -e HATRIX_err.log

source ~/.bashrc

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_BUILD_TYPE=Release
make -j all

for N in 1024 2048 4096 8192 16384 32768 65536 131072; do
    for matrix_type in 1; do
        for admis in 0; do
            for rank in 5 10 20 40 70 80; do
                ./examples/UMV_H2_far_dense $N $rank 128 $admis 1 diagonal_admis 0 $matrix_type
            done
        done
    done
done
