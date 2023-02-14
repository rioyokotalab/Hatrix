#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -N HATRIX
#$ -o HATRIX_out.log
#$ -e HATRIX_err.log

source ~/.bashrc

module purge
module load intel-mpi gcc intel-mkl cmake/3.22.3 intel-vtune intel-itac

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64/pkgconfig:/home/acb10922qh/gitrepos/googletest/build/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64

# put in bash_profile for remote MPI processes.
# ulimit -c unlimited             # does not pass to remote child processes.

export MKL_NUM_THREADS=40
export OMP_NUM_THREADS=40
export OMP_PLACES=cores
export OMP_PROC_BIND=close

make -j H2_main

nleaf=1024
ndim=3

# BLR2
for adm in 1; do
    for pert in 0; do
        for max_rank in 100; do
            for N in 65536; do
                for i in `seq 1`; do
                    ./bin/H2_main --N $N \
                                  --nleaf $nleaf \
                                  --kernel_func laplace \
                                  --kind_of_geometry grid \
                                  --ndim $ndim \
                                  --max_rank $max_rank \
                                  --accuracy -1 \
                                  --admis $adm \
                                  --admis_kind geometry \
                                  --construct_algorithm miro \
                                  --add_diag 1e-9 \
                                  --kind_of_recompression 3 \
                                  --perturbation $pert
                done
            done
        done
    done
done

# H2
for adm in 1; do
    for pert in 0; do
        for max_rank in 100; do
            for N in 65536; do
                for i in `seq 1`; do
                    ./bin/H2_main --N $N \
                                  --nleaf $nleaf \
                                  --kernel_func laplace \
                                  --kind_of_geometry grid \
                                  --ndim $ndim \
                                  --max_rank $max_rank \
                                  --accuracy -1 \
                                  --admis $adm \
                                  --admis_kind geometry \
                                  --construct_algorithm miro \
                                  --add_diag 1e-9 \
                                  --kind_of_recompression 3 \
                                  --perturbation $pert \
                                  --use_nested_basis
                done
            done
        done
    done
done
