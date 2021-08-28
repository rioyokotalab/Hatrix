#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J hss
#SBATCH --time=10:00:00

make clean
make -j all

FILE=blr_qsparse.csv


./bin/Qsparse_weak_1level 40 10 10 $FILE

# for N in 1000 2000 4000 8000 16000 32000 64000; do
#     for block in 50 100 200 500 1000; do
#         for rank in 3 10 20 40 80 100 300 400 500; do
#             ./bin/Qsparse_weak_1level $N $rank $block $FILE
#         done
#     done
# done
