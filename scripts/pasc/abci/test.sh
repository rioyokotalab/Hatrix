#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=12:00:00
#$ -N first_test
#$ -o first_test_out.log
#$ -e first_test_err.log

source ~/.bashrc

module purge
module load intel-mpi/2021.5 gcc/11.2.0 intel-mkl/2022.0.0 cmake/3.22.3

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64/pkgconfig:/home/acb10922qh/gitrepos/googletest/build/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64

export OMP_PLACES=cores

make -j H2_dtd
# mpirun -n $procs -ppn 2 -f $SGE_JOB_HOSTLIST
