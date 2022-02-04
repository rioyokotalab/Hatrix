#!/bin/bash
#SBATCH --job-name=BLR2_FAR_TEST      # Job name
#SBATCH --nodes=1                     # Run on a single CPU
#SBATCH --time=10:00:00               # Time limit hrs:min:sec

module load intel-mkl/2020.4.304/gcc-7.3.0-52gb
# module load valgrind/3.15.0/gcc-7.3.0-74vd

# mkdir build
# cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$PWD
# make -j
# rm *out
make -j all
# make UMV_strong_H2_Nlevel_starsh


./bin/UMV_BLR2_far_dense 100 20 4 2 2 diagonal_admis
# echo -n "\n"
./bin/UMV_BLR2_far_dense 500 100 4 0.8 2 geometry_admis
./bin/UMV_BLR2_far_dense 500 100 10 0.8 2 geometry_admis
# ./bin/UMV_BLR2_far_dense 500 50 4 0.3 2 geometry_admis
# echo -n "\n"
# ./bin/UMV_BLR2_far_dense 500 50 8 0.3 2 geometry_admis
# echo -n "\n"
# ./bin/UMV_BLR2_far_dense 500 50 12 0.3 2 geometry_admis


# ./bin/UMV_strong_chained_product 60 4 5 1
