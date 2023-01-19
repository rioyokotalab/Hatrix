#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -N HATRIX
#$ -o HATRIX_out.log
#$ -e HATRIX_err.log

source ~/.bashrc

module purge
module load intel-mpi/2021.5 gcc intel-mkl/2022.0.0 cmake/3.22.3 intel-vtune intel-itac

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64/pkgconfig:/home/acb10922qh/gitrepos/googletest/build/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64

# put in bash_profile for remote MPI processes.
# ulimit -c unlimited             # does not pass to remote child processes.

export OMP_PLACES=cores
# export OMP_NUM_THREADS=1
# mpirun -n 4 -genv I_MPI_DEBUG=10  xterm -e gdb -ex=run --args ./bin/H2_dtd --N $N \
# mpirun -n 1  -gtool "gdb:0=attach" ./bin/H2_dtd --N $N \
# mpirun -n 1 -gtool "gdb:0=attach" ./bin/H2_dtd --N $N \

# mpiexec.hydra -n 2 -genv I_MPI_BIND_NUMA=0,1 xterm -e gdb -ex=run --args ./bin/H2_dtd --N $N \
make -j H2_dtd VERBOSE=1

# export GMON_OUT_PREFIX=gmon.out-

for adm in 0.8; do
    nleaf=256
    ndim=3
    max_rank=150

    for N in 8192; do

        mpirun -n 2 \
               ./bin/H2_dtd --N $N \
               --nleaf $nleaf \
               --kernel_func laplace \
               --kind_of_geometry grid \
               --ndim $ndim \
               --max_rank $max_rank \
               --accuracy 1e-8 \
               --admis $adm \
               --admis_kind geometry \
               --construct_algorithm miro \
               --add_diag 1e-10 \
               --use_nested_basis
    done
done

gprof -s bin/H2_dtd gmon.out-*
gprof -q bin/H2_dtd gmon.sum > call_graph.out
gprof bin/H2_dtd gmon.sum > gprof_out.out
