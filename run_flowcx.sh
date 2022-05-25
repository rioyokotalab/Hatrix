#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=cx-small"
#PJM -L "elapse=86400"
#PJM --mpi "proc=1"
#PJM -j

module purge
module load gcc/8.4.0 openmpi cmake
source /home/center/opt/x86_64/cores/intel/compilers_and_libraries_2020.4.304/linux/mkl/bin/mklvars.sh intel64

export I_MPI_CXX=g++

make clean
make -j HSS_scalapack

for nprocs in 2; do
    mpirun -n $nprocs  ./bin/HSS_scalapack --N 2048 \
           --nleaf 256 \
           --kernel_func laplace \
           --kind_of_geometry circular \
           --ndim 1 \
           --max_rank 100 \
           --accuracy 1e-11 \
           --admis 0 \
           --admis_kind diagonal \
           --construct_algorithm miro \
           --add_diag 1e-5 \
           --use_nested_basis
done
