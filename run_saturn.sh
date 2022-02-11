#!/bin/bash
#SBATCH --job-name=BLR2_FAR_TEST      # Job name
#SBATCH --nodes=1                     # Run on a single CPU
#SBATCH --time=10:00:00               # Time limit hrs:min:sec

source ~/.bashrc

module load intel-mkl/2020.4.304/gcc-7.3.0-52gb
# module load valgrind/3.15.0/gcc-7.3.0-74vd


# make clean
make -j all
# gdb -ex run --args ./bin/H2_far_dense_construct 400 5 80 0.7 2 geometry_admis
# ./bin/UMV_BLR2_far_dense 200 40 15 1 2 geometry_admis
# ./bin/UMV_BLR2_far_dense 400 40 15 1.3 2 geometry_admis
for rank in 5 6 7 8 10; do
    ./bin/UMV_BLR2_far_dense 4000 100 $rank 0.2 2 geometry_admis
done


# gdb -ex run  --args ./bin/UMV_BLR2_far_dense 400 80 15 0.7 2 geometry_admis
