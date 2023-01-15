#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J VALGRIND
#SBATCH --time=72:00:00

source /etc/profile.d/modules.sh
module purge
module load cuda intel/2022/mkl gcc/8.4 cmake lapack/3.9.0 openmpi/4.0.5

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig

#:/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64::/home/sameer.deshmukh/gitrepos/papi/src/lib

# :/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/release

make -j H2_dtd

for adm in 0.8; do
    nleaf=1024
    ndim=3
    max_rank=50

    for N in 65536; do
        mpirun -n 1 ./bin/H2_dtd --N $N \
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
                      --use_nested_basis \
                      -- \
                      --mca profile_filename demo_parsec_profile \
                      --mca mca_pins task_profiler
    done
done
