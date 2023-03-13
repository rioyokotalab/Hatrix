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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64

# put in bash_profile for remote MPI processes.
# ulimit -c unlimited             # does not pass to remote child processes.

# export MKL_NUM_THREADS=40
# export OMP_NUM_THREADS=40
export OMP_PLACES=cores
export OMP_PROC_BIND=close

make -j H2_dtd
# make -j H2_main

for adm in 1; do
    nleaf=32
    ndim=3
    max_rank=20

    # gdb -q -iex "set auto-load safe-path /home/user/gdb" -ex run --args

    for N in 65536; do
        mpirun -l -n 64 ./bin/H2_dtd --N $N \
                      --nleaf $nleaf \
                      --kernel_func laplace \
                      --kind_of_geometry grid \
                      --ndim $ndim \
                      --max_rank $max_rank \
                      --accuracy -1 \
                      --admis $adm \
                      --admis_kind diagonal \
                      --construct_algorithm miro \
                      --add_diag 1e-9  \
                      --kind_of_recompression 3
    done
done
