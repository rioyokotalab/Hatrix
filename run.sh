#!/bin/bash
#YBATCH -r dgx-a100_2
#SBATCH -N 1
#SBATCH -J UMV
#SBATCH --time=12:00:00

make clean
make -j all

FILE=test_hatrix.csv

./bin/UMV_weak_1level 1000 10 100 $FILE
cat $FILE

# for N in 1000 5000 10000 20000 40000 80000; do
#     for rank in 5 10 20 40 60 100; do
#         for block in 100 500 1000 2000 4000; do
#             ./bin/UMV_weak_1level $N $rank $block $FILE
#         done
#     done
# done
