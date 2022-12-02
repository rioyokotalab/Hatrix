#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=small"
#PJM -L "elapse=10:00:00"
#PJM --llio cn-cache-size=1Gi
#PJM --llio sio-read-cache=on
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -x PJM_LLIO_GFSCACHE=/vol0003
#PJM -s

source /vol0004/apps/oss/spack/share/spack/setup-env.sh

source ~/.bashrc

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/hp190122/u01594/gitrepos/googletest/build/lib64/pkgconfig:/vol0003/hp190122/u01594/gitrepos/lorapo/stars-h-rio/build/installdir/lib/pkgconfig
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/vol0003/hp190122/u01594/gitrepos/parsec/build/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol0003/hp190122/u01594/gitrepos/parsec/build/lib64

make -j H2_main
make -j H2_dtd


for adm in 0.8; do
    nleaf=512
    ndim=2
    max_rank=110

    for N in 4096; do
	    echo "running"
        mpirun -n 1 ./bin/H2_dtd --N $N \
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
