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

make -j H2_main

for adm in 0.8; do
    ./bin/H2_main --N 512 \
         --nleaf 128 \
         --kernel_func laplace \
         --kind_of_geometry circular \
         --ndim 2 \
         --max_rank 50 \
         --accuracy 1e-11 \
         --admis $adm \
         --admis_kind geometry \
         --construct_algorithm miro \
         --add_diag 1e-7 \
         --use_nested_basis
done

