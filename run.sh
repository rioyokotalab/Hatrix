#!/bin/bash
#YBATCH -r dgx-a100_2
#SBATCH -N 1
#SBATCH -J HSS
#SBATCH --time=12:00:00

make clean
make -j all

FILE=test_hatrix.csv

rm $FILE
./bin/HSS_2level_construct 1000 10 100 $FILE
./bin/HSS_2level_construct 1000 100 100 $FILE

cat $FILE
