#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J VALGRIND
#SBATCH --time=72:00:00

source ~/.bashrc
source /home/sameer.deshmukh/gitrepos/parsec/build/bin/parsec.env.sh

source /etc/profile.d/modules.sh
module purge
module load cuda intel/2022/mkl gcc/8.4 cmake lapack/3.9.0 openmpi/4.0.5

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig

#:/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64::/home/sameer.deshmukh/gitrepos/papi/src/lib

# :/mnt/nfs/packages/x86_64/intel/2022/mpi/2021.6.0/lib/release

make -j H2_dtd

for adm in 0.8; do
    ndim=3
    N=65536

    for max_rank in 50 100 150 200; do
        for nleaf in 512 1024 2048 4096 8192; do
            mpirun --mca opal_warn_on_missing_libcuda 0 \
                   -n 1 ./bin/H2_dtd --N $N \
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

            profile2h5 test_profile_output-0.prof
            python hdf_read.py
            rm -rf test_profile_output-0*
        done

    done
done
