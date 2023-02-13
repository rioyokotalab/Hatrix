#!/bin/bash
#YBATCH -r dgx-a100_4
#SBATCH -N 1
#SBATCH -J PAPI
#SBATCH --time=72:00:00

source ~/.bashrc
source /home/sameer.deshmukh/gitrepos/parsec/build/bin/parsec.env.sh

source /etc/profile.d/modules.sh
module purge
module load cuda intel/2022/mkl gcc cmake lapack/3.9.0 openmpi/4.0.5

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig

#:/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64::/home/sameer.deshmukh/gitrepos/papi/src/lib

make -j H2_main

for adm in 0.7 0.8 0.9 1.2 1.4 1.6 1.8; do
    nleaf=512
    max_rank=50
    ndim=3

    for N in 131072; do
        ./bin/H2_main --N $N \
               --nleaf $nleaf \
               --kernel_func laplace \
               --kind_of_geometry grid \
               --ndim $ndim \
               --max_rank $max_rank \
               --accuracy 1e-12 \
               --admis $adm \
               --admis_kind geometry \
               --construct_algorithm miro \
               --add_diag 1e-9 \
               --kind_of_recompression 3 \
               --use_nested_basis

    done
done
