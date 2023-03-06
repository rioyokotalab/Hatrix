#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscunit=rscunit_ft01"
#PJM -L "rscgrp=small"
#PJM -L "elapse=24:00:00"
#PJM -L "freq=2200"
#PJM -L "throttling_state=0"
#PJM -L "issue_state=0"
#PJM -L "ex_pipe_state=0"
#PJM --mpi "proc=1"
#PJM --mpi "max-proc-per-node=1"
#PJM -s

# source /vol0004/apps/oss/spack/share/spack/setup-env.sh

#source ~/.bashrc

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/hp190122/u01594/gitrepos/googletest/build/lib64/pkgconfig:/vol0003/hp190122/u01594/gitrepos/lorapo/stars-h-rio/build/installdir/lib/pkgconfig
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/hp190122/u01594/gitrepos/parsec/build/lib64/pkgconfig:/vol0003/hp190122/u01594/gitrepos/parsec/build/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol0003/hp190122/u01594/gitrepos/parsec/build/lib64

#export PARALLEL=1
#export OMP_NUM_THREADS=1
# export PLE_MPI_STD_EMPTYFILE=off
# export FLIB_SCCR_CNTL=FALSE
# export FLIB_PTHREAD=1

export OMP_PLACES=cores
export OMP_DISPLAY_AFFINITY=TRUE
export OMP_PROC_BIND=close
export OMP_BIND=close
#export OMP_NUM_THREADS=1
export XOS_MMM_L_PAGING_POLICY="demand:demand:demand"

# make -j H2_main
#make clean
make -j H2_dtd
# make -j H2_main


for adm in 1; do
    nleaf=256
    ndim=3

    for max_rank in 50; do
        for N in 8192; do
            for i in `seq 1`; do
                ./bin/H2_dtd --N $N \
                              --nleaf $nleaf \
                              --kernel_func laplace \
                              --kind_of_geometry grid \
                              --ndim $ndim \
                              --max_rank $max_rank \
                              --accuracy -1 \
                              --admis $adm \
                              --admis_kind diagonal \
                              --construct_algorithm miro \
                              --add_diag 1e-9 \
                              --kind_of_recompression 3 \
                              --use_nested_basis
            done
        done
    done
done
