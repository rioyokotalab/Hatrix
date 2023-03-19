#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J SAMEH
#SBATCH --time=24:00:00

source ~/.bashrc
source /home/sameer.deshmukh/gitrepos/parsec/build/bin/parsec.env.sh

source /etc/profile.d/modules.sh
module purge
module load cuda intel/2022/mkl gcc/10.4 cmake lapack/3.9.0 openmpi/4.0.5

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/gsl-2.7.1/build/lib/pkgconfig

#:/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64:/home/sameer.deshmukh/gitrepos/papi/src/lib:/home/sameer.deshmukh/gitrepos/gsl-2.7.1/build/lib

# export MKL_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close


make -j H2_main

nleaf=256
max_rank=50
ndim=2
adm=0

for N in 40000; do
    for p1 in 1e-2 1e-3 1e-4 1e-5; do
        for p2 0.5 0.8 1; do
            for p3 in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
                ./bin/H2_main --N $N \
                              --nleaf $nleaf \
                              --kernel_func gsl_matern \
                              --kind_of_geometry grid \
                              --ndim $ndim \
                              --max_rank $max_rank \
                              --accuracy -1 \
                              --admis $adm \
                              --admis_kind diagonal \
                              --construct_algorithm miro \
                              --geometry_file "/home/sameer.deshmukh/XY_40000_1" \
                              --param_1 $p1 --param_2 $p2 --param_3 $p3  \
                              --kind_of_recompression 3
            done
        done
    done
done
