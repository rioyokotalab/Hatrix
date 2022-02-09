#!/bin/bash
#SBATCH --job-name=BLR2_FAR_TEST      # Job name
#SBATCH --nodes=1                     # Run on a single CPU
#SBATCH --time=10:00:00               # Time limit hrs:min:sec

source ~/.bashrc

module load intel-mkl/2020.4.304/gcc-7.3.0-52gb

# gtest pkg-config
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/home/v0dro/gitrepos/Hatrix/dependencies/GTest/lib64/pkgconfig"

# stars-h pkg-config
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/home/v0dro/gitrepos/stars-h/build/lib/pkgconfig"


# make clean
make -j starsh_programs

for admis in 0.1 0.2 0.4 0.5 0.7 0.8 1 1.2; do
    ./bin/UMV_BLR2_far_dense_starsh 4000 400 50 $admis 2 geometry_admis 2
done

# ./bin/UMV_strong_1level_starsh 1000 10 10 1 2
