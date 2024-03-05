#!/bin/bash
#PJM -L "node=128"
#PJM -L "rscunit=rscunit_ft01"
#PJM -L "rscgrp=small"
#PJM -L "elapse=24:00:00"
#PJM --mpi "proc=128"
#PJM --mpi "max-proc-per-node=1"
#PJM -s

#source ~/.bashrc
set -e

. /vol0004/apps/oss/spack/share/spack/setup-env.sh
spack load /wyds2me

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/hp190122/u01594/gitrepos/googletest/build/lib64/pkgconfig:/vol0003/hp190122/u01594/gitrepos/lorapo/stars-h-rio/build/installdir/lib/pkgconfig
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/hp190122/u01594/gitrepos/parsec/build/lib64/pkgconfig:/vol0003/hp190122/u01594/gitrepos/parsec/build/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64:/vol0003/hp190122/u01594/gitrepos/parsec/build/lib64:/home/hp190122/u01594/gsl-2.7.1/build/lib

# export PARALLEL=1
export OMP_NUM_THREADS=48
export PLE_MPI_STD_EMPTYFILE=off
export FLIB_SCCR_CNTL=FALSE
# export FLIB_PTHREAD=1

export OMP_PLACES=cores
export OMP_DISPLAY_AFFINITY=TRUE
export OMP_PROC_BIND=close
export OMP_BIND=close
export XOS_MMM_L_PAGING_POLICY="demand:demand:demand"


mkdir -p build

pushd build
cmake .. -DCMAKE_CXX_FLAGS=" -Nclang -std=c++17 "  \
      -DGSL_INCLUDE_DIR="/home/hp190122/u01594/gsl-2.7.1/build/include" \
      -DGSL_LIBRARY="/home/hp190122/u01594/gsl-2.7.1/build/lib/libgsl.a" \
      -DGSL_CBLAS_LIBRARY="/home/hp190122/u01594/gsl-2.7.1/build/lib/libgslcblas.a" \
      -DHatrix_BUILD_TESTS=0 \
      -DCMAKE_BUILD_TYPE=Debug
make -j VERBOSE=1
popd

for N in 512 2048; do
    ./build/examples/H2_strong_CON_sameer \
        $N 64 0 40 0.5 0 1 3 1
done
