#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J UMV
#SBATCH --time=100:00:00

make clean
make -j all

FILE=blr_qsparse.csv

for N in 1000; do
    for rank in 100; do
        gdb -ex run --args ./bin/UMV_weak_1level 1000 $rank 100 $FILE
    done
done



# for N in 1000 5000 10000 20000 40000 80000; do
#     for rank in 5 10 20 40 60 100; do
#         for block in 100 500 1000 2000 4000; do
#             ./bin/UMV_weak_1level $N $rank $block $FILE
#         done
#     done
# done
