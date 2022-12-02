#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J GPROF
#SBATCH --time=72:00:00

source /etc/profile.d/modules.sh
module purge
# module load intel/2022/mkl cmake lapack/3.9.0 openmpi/4.0.5
module load intel/2022/mkl cmake lapack/3.9.0 openmpi

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib

# :/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/release

# rm -rf build
# mkdir build
# cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_BUILD_TYPE=Debug \
#       -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg
# make -j all

make -j H2_main
make -j H2_dtd
for adm in 0.8; do
    nleaf=512
    ndim=2
    max_rank=110
    for N in 4096; do
        mpirun --mca opal_warn_on_missing_libcuda 0 \
               -np 2 xterm -e gdb --args ./bin/H2_dtd --N $N \
                     --nleaf $nleaf \
                     --kernel_func laplace \
                     --kind_of_geometry grid \
                     --ndim $ndim \
                     --max_rank $max_rank \
                     --accuracy 1e-11 \
                     --admis $adm \
                     --admis_kind geometry \
                     --construct_algorithm miro \
                     --add_diag 1e-9 \
                     --use_nested_basis
    done
done
