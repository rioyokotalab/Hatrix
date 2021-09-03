#!/bin/bash
#YBATCH -r dgx-a100_2
#SBATCH -N 1
#SBATCH -J HSS
#SBATCH --time=12:00:00

make clean
make -j all

FILE=test_hatrix.csv

rm $FILE


for N in 100 1000; do
    for rank in 5 10 20 25 40 100; do
        ./bin/HSS_2level_construct $N $rank
    done
done
