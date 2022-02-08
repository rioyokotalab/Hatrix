#!/bin/bash
#SBATCH --job-name=BLR2_FAR_TEST      # Job name
#SBATCH --nodes=1                     # Run on a single CPU
#SBATCH --time=10:00:00               # Time limit hrs:min:sec

source ~/.bashrc

module load intel-mkl/2020.4.304/gcc-7.3.0-52gb


# mkdir build
# cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$PWD
# make -j
# rm *out
# make -j all
# make -j UMV_strong_H2_Nlevel_starsh

# make clean
make -j starsh_programs

./UMV_strong_1level_starsh 1000 10 10 1 1
