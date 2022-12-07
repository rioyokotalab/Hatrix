#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J GPROF
#SBATCH --time=72:00:00

source /etc/profile.d/modules.sh
module purge
module load cuda intel/2022/mkl gcc/8.4 cmake lapack/3.9.0 openmpi/4.0.5

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig

#:/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64::/home/sameer.deshmukh/gitrepos/papi/src/lib

# :/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/release

make -j H2_main
# make -j H2_dtd VERBOSE=1                  #
for adm in 0.4 0.8 0.81 0.82 0.83 0.84 0.85 0.88 0.89 0.91 0.92 0.95 0.96 0.98 1 1.05 1.1; do
    nleaf=1024
    ndim=3
    max_rank=400
    for N in 16384 32768 65536 131072; do
        ./bin/H2_main --N $N \
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
