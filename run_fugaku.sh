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
