#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J PAPI
#SBATCH --time=72:00:00

source ~/.bashrc
source /home/sameer.deshmukh/gitrepos/parsec/build/bin/parsec.env.sh

source /etc/profile.d/modules.sh
module purge
module load cuda intel/2022/mkl gcc/8.4 cmake lapack/3.9.0 openmpi/4.0.5

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64::/home/sameer.deshmukh/gitrepos/papi/src/lib

make -j H2_dtd

for adm in 7; do
    nleaf=4096

    for max_rank in 4000; do
        for ndim in 2; do
            for N in 32768; do
                mpirun --mca opal_warn_on_missing_libcuda 0 -n 1 \
                       ./bin/H2_dtd --N $N \
                       --nleaf $nleaf \
                       --kernel_func laplace \
                       --kind_of_geometry circular \
                       --ndim $ndim \
                       --max_rank $max_rank \
                       --accuracy 1e-14 \
                       --qr_accuracy 1e-6 \
                       --kind_of_recompression 0 \
                       --admis $adm \
                       --admis_kind geometry \
                       --construct_algorithm miro \
                       --add_diag 1e-8 \
                       --use_nested_basis
            done
        done
    done
done
