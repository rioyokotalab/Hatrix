#!/bin/bash
#SBATCH --job-name=BLR2_FAR_TEST      # Job name
#SBATCH --nodes=1                     # Run on a single CPU
#SBATCH --time=10:00:00               # Time limit hrs:min:sec

source ~/.bashrc

module load intel-mkl/2020.4.304/gcc-7.3.0-52gb
# module load valgrind/3.15.0/gcc-7.3.0-74vd

# mkdir build
# cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$PWD
# make -j
# rm *out
# make -j all
# make -j UMV_strong_H2_Nlevel_starsh

# for rank in 5 10 15; do
#     ./bin/UMV_strong_H2_Nlevel_starsh 1600 $rank 5 2 3
# done


make UMV_strong_H2_Nlevel_starsh
# ./bin/UMV_strong_H2_Nlevel 800 5 4 1
# ./bin/UMV_strong_H2_Nlevel 800 5 4 2
# ./bin/UMV_strong_H2_Nlevel 400 6 3 2
# ./bin/UMV_strong_H2_Nlevel 1600 5 5 1


for rank in 6 10 15 20; do
    for admis in 1 2 3; do
        ./bin/UMV_strong_H2_Nlevel_starsh 800 $rank  3 $admis 2
        ./bin/UMV_strong_H2_Nlevel_starsh 1600 $rank 4 $admis 2
        ./bin/UMV_strong_H2_Nlevel_starsh 3200 $rank 5 $admis 2
        ./bin/UMV_strong_H2_Nlevel_starsh 6400 $rank 6 $admis 2
    done
done
