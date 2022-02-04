#!/bin/bash
#SBATCH --job-name=HATRIX      # Job name
#SBATCH --nodes=1                     # Run on a single CPU
#SBATCH --time=10:00:00               # Time limit hrs:min:sec

module load intel-mkl/2020.4.304/gcc-7.3.0-52gb
# module load valgrind/3.15.0/gcc-7.3.0-74vd

# mkdir build
# cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$PWD
# make -j
make -j all
# make UMV_strong_H2_Nlevel_starsh

# ./examples/UMV_strong_1level 40 4 4 0
./bin/UMV_BLR2_far_dense 40 10 4 2 2 diagonal_admis
./bin/UMV_BLR2_far_dense 100 20 4 2 2 diagonal_admis
./bin/UMV_BLR2_far_dense 800 100 10 2 2 diagonal_admis

# ./bin/UMV_strong_chained_product 60 4 5 1
