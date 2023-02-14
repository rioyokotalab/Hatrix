#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=2:00:00
#$ -N H2
#$ -o H2_out.log
#$ -e H2_err.log

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
max_rank=100

for i in "16384 1 0" "65536 1 0"; do
    set -- $i
    N=$1
    adm=$2
    pert=$3

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
                  --perturbation $pert \
                  --use_nested_basis

done

# gprof -s bin/H2_dtd gmon.out-*
# gprof -q bin/H2_dtd gmon.sum > call_graph.out
# gprof bin/H2_dtd gmon.sum > gprof_out.out
