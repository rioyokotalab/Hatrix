#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=cx-"
#PJM -L "elapse=86400"
#PJM --mpi "proc=2"
#PJM -j

module purge
module load gcc/8.4.0 openmpi cmake
source /home/center/opt/x86_64/cores/intel/compilers_and_libraries_2020.4.304/linux/mkl/bin intel64

export I_MPI_CXX=g++



make -j HSS_scalapack
