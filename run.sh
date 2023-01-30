#!/bin/bash
#YBATCH -r epyc-7502_4
#SBATCH -N 1
#SBATCH -J PAPI
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
# mpirun --mca opal_warn_on_missing_libcuda 0 \

make -j H2_dtd

for adm in 2; do
    nleaf=512
    max_rank=30

    for ndim in 1; do
        for N in 8192; do
            mpirun --mca opal_warn_on_missing_libcuda 0 -n 4 \
                   xterm -e gdb --args ./bin/H2_dtd --N $N \
                          --nleaf $nleaf \
                          --kernel_func laplace \
                          --kind_of_geometry circular \
                          --ndim $ndim \
                          --max_rank $max_rank \
                          --accuracy 1e-8 \
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

    # for kind_of_recompression in 0 1 2 3; do
    #     for ndim in 2 3; do
    #         for N in 8192 16384 32768 65536 131072; do
    #             ./bin/H2_main --N $N \
    #                           --nleaf $nleaf \
    #                           --kernel_func laplace \
    #                           --kind_of_geometry circular \
    #                           --ndim $ndim \
    #                           --max_rank $max_rank \
    #                           --accuracy 1e-8 \
    #                           --qr_accuracy 1e-6 \
    #                           --kind_of_recompression $kind_of_recompression \
    #                           --admis $adm \
    #                           --admis_kind geometry \
    #                           --construct_algorithm miro \
    #                           --add_diag 1e-8 \
    #                           --use_nested_basis


    #         done
    #     done
    # done

# file_name=${N}_${nleaf}_${max_rank}_task_profile.prof

# mv test_profile_output-0.prof $file_name
# profile2h5 $file_name
# python hdf_read.py
# rm -rf test_profile_output-0*
