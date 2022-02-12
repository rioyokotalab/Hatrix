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
# ./bin/UMV_BLR2_far_dense 500 100 7 0.4 2 geometry_admis
./bin/UMV_BLR2_far_dense 500 100 3 0.6 2 geometry_admis
# ./bin/UMV_BLR2_far_dense 2000 100 7 0.4 2 geometry_admis


# ./bin/UMV_BLR2_far_dense 250 50 7 0.6 2 geometry_admis
# ./bin/UMV_BLR2_far_dense 250 50 7 1 2 diagonal_admis
# echo "2D CIRCLE"
# echo "ADMIS 0.4"
# for rank in 5 7 10 12 20; do
#     ./bin/UMV_BLR2_far_dense 4000 100 $rank 0.4 2 geometry_admis
# done

# echo "ADMIS 0.2"
# for rank in 5 7 10 12 20; do
#     ./bin/UMV_BLR2_far_dense 4000 100 $rank 0.2 2 geometry_admis
# done

# echo "3D SPHERE"
# echo "ADMIS 0.4"
# for rank in 10 12 15 18 20; do
#     ./bin/UMV_BLR2_far_dense 4000 100 $rank 0.4 3 geometry_admis
# done

# echo "ADMIS 0.2"
# for rank in 10 12 15 18 20; do
#     ./bin/UMV_BLR2_far_dense 4000 100 $rank 0.2 3 geometry_admis
# done


# gdb -ex run  --args ./bin/UMV_BLR2_far_dense 400 80 15 0.7 2 geometry_admis
