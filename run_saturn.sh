#!/bin/bash
#SBATCH --job-name=HATRIX      # Job name
#SBATCH --nodes=1                     # Run on a single CPU
#SBATCH --time=10:00:00               # Time limit hrs:min:sec

module load intel-mkl/2020.4.304/gcc-7.3.0-52gb
module load valgrind/3.15.0/gcc-7.3.0-74vd

# mkdir build
# cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$PWD
# make -j
make UMV_strong_H2_Nlevel_starsh

# ./examples/UMV_strong_1level 40 4 4 0
