#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=2:00:00
#$ -N H2
#$ -o H2_out.log
#$ -e H2_err.log

source ~/.bashrc

module purge
module load intel-mpi gcc intel-mkl cmake/3.22.3 intel-vtune intel-itac

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64/pkgconfig:/home/acb10922qh/gitrepos/googletest/build/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64:/home/acb10922qh/gsl-2.7.1/build/lib

# put in bash_profile for remote MPI processes.
# ulimit -c unlimited             # does not pass to remote child processes.

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close


make -j H2_main

nleaf=256
max_rank=50
ndim=1
adm=0

# valgrind --leak-check=full --track-origins=yes

for N in 1024; do
    for adm in 1; do
        for nleaf in 128; do
            for max_rank in 50; do
                ./bin/H2_main --N $N \
                              --nleaf $nleaf \
                              --kernel_func gsl_matern \
                              --kind_of_geometry grid \
                              --ndim $ndim \
                              --max_rank $max_rank \
                              --accuracy -1 \
                              --admis $adm \
                              --admis_kind geometry \
                              --construct_algorithm miro \
                              --param_1 1 --param_2 0.03 --param_3 0.5 \
                              --kind_of_recompression 3
            done
        done
    done
done
