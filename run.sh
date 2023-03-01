#!/bin/bash
#YBATCH -r dgx-a100_8
#SBATCH -N 1
#SBATCH -J BLR2
#SBATCH --time=72:00:00

source ~/.bashrc
source /home/sameer.deshmukh/gitrepos/parsec/build/bin/parsec.env.sh

source /etc/profile.d/modules.sh
module purge
module load cuda intel/2022/mkl gcc/10.4 cmake lapack/3.9.0 openmpi/4.0.5

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig

#:/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64:/home/sameer.deshmukh/gitrepos/papi/src/lib

# export MKL_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close


make -j H2_dtd
# make -j H2_main

for adm in 1; do
    nleaf=64
    ndim=3
    max_rank=32

    for N in 16384; do
        mpirun -n 1 ./bin/H2_dtd --N $N \
                      --nleaf $nleaf \
                      --kernel_func laplace \
                      --kind_of_geometry grid \
                      --ndim $ndim \
                      --max_rank $max_rank \
                      --accuracy -1 \
                      --admis $adm \
                      --admis_kind diagonal \
                      --construct_algorithm miro \
                      --add_diag 1e-9  \
                      --kind_of_recompression 3 \
                      --use_nested_basis
    done
done
