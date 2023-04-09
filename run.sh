#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J H2_gsl
#SBATCH --time=24:00:00

source ~/.bashrc
# source /home/sameer.deshmukh/gitrepos/parsec/build/bin/parsec.env.sh

source /etc/profile.d/modules.sh
module purge
module load gcc/12.2 cuda intel/2022/mkl cmake intel/2022/mpi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gsl-2.7.1/build/lib

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64:/home/sameer.deshmukh/gitrepos/papi/src/lib:/home/sameer.deshmukh/gitrepos/gsl-2.7.1/build/lib

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# ROOT=$PWD
# cd examples/distributed/H2_ptg
# ./compile_jdf.sh
# cd $ROOT

make -j H2_dtd
make -j H2_main

ndim=2

# ./build/examples/UMV_H2_Nlevel 64 16 0 10 60 1.9 0 2 2 0

for N in 64; do
    for adm in 0; do
        for nleaf in 16; do
            for max_rank in 10; do
                mpirun -n 2 ./bin/H2_dtd --N $N \
                              --nleaf $nleaf \
                              --kernel_func gsl_matern \
                              --kind_of_geometry grid \
                              --ndim $ndim \
                              --max_rank $max_rank \
                              --accuracy -1 \
                              --admis $adm \
                              --admis_kind diagonal \
                              --construct_algorithm miro \
                              --param_1 1e-9 --param_2 0.03 --param_3 0.5 \
                              --kind_of_recompression 3 --use_nested_basis
            done
        done
    done
done


# for N in 16384; do
#     for adm in 0.1; do
#         for nleaf in 256; do
#             for max_rank in 200; do
#                 ./bin/H2_main --N $N \
#                               --nleaf $nleaf \
#                               --kernel_func gsl_matern \
#                               --kind_of_geometry grid \
#                               --ndim $ndim \
#                               --max_rank $max_rank \
#                               --accuracy -1 \
#                               --admis $adm \
#                               --admis_kind diagonal \
#                               --construct_algorithm miro \
#                               --param_1 1 --param_2 0.03 --param_3 0.5 \
#                               --kind_of_recompression 3 --use_nested_basis
#             done
#         done
#     done
# done

# for N in 16384; do
#     for adm in 0.1; do
#         for nleaf in 256; do
#             for max_rank in 150; do
#                 ./bin/H2_main --N $N \
#                               --nleaf $nleaf \
#                               --kernel_func yukawa \
#                               --kind_of_geometry grid \
#                               --ndim $ndim \
#                               --max_rank $max_rank \
#                               --accuracy -1 \
#                               --admis $adm \
#                               --admis_kind geometry \
#                               --construct_algorithm miro \
#                               --param_1 1 --param_2 1e-9 \
#                               --kind_of_recompression 3 --use_nested_basis
#             done
#         done
#     done
# done
