#!/bin/bash
#PJM -L "node=4"
#PJM -L "rscgrp=cx-small"
#PJM -L "elapse=2:00:00"
#PJM --mpi "proc=8"
#PJM -j

module purge
module load gcc/8.4.0 openmpi cmake
source /home/center/opt/x86_64/cores/intel/compilers_and_libraries_2020.4.304/linux/mkl/bin/mklvars.sh intel64

export I_MPI_CXX=g++

make clean
make -j HSS_scalapack

# for nprocs in 4; do
mpirun -n $PJM_MPI_PROC -machinefile $PJF_O_NODEINF -npernode 2  \
       ./bin/HSS_scalapack --N 262144 \
       --nleaf 2048 \
       --kernel_func laplace \
       --kind_of_geometry grid \
       --ndim 1 \
       --max_rank 1000 \
       --accuracy 1e-11 \
       --admis 0 \
       --admis_kind diagonal \
       --construct_algorithm miro \
       --add_diag 1e-8 \
       --use_nested_basis
# done
