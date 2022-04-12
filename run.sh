#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J GPROF
#SBATCH --time=72:00:00

source ~/.bashrc

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg
make -j all

# 32768
for N in 16384; do
    for matrix_type in 1; do
        for admis in 1000; do
            for rank in 20; do
                ./examples/UMV_H2_far_dense $N $rank 128 $admis 3 geometry_admis 0 $matrix_type
            done
        done
    done
done
