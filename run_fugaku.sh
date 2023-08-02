#!/bin/bash
#PJM -L "node=128"
#PJM -L "rscunit=rscunit_ft01"
#PJM -L "rscgrp=small"
#PJM -L "elapse=24:00:00"
#PJM -L "freq=2200"
#PJM -L "throttling_state=0"
#PJM -L "issue_state=0"
#PJM -L "ex_pipe_state=0"
#PJM --mpi "proc=128"
#PJM --mpi "max-proc-per-node=1"
#PJM -s

#source ~/.bashrc
set -e

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

# Generate the points for the ELSES matrix.
export ELSES_ROOT=/data/hp190122/users/u01594/elses
mol_folder=$ELSES_ROOT/sample/sample_non_geno/C60_fcc2x2x2_disorder_expand_1x1x1
source_file=$mol_folder/C60_fcc2x2x2_20220727.xyz
fcc_xml_file=C60_fcc.xml

exec_supercell=$ELSES_ROOT/make_supercell_C60_FCCs_w_noise/a.out
exec_elses_xml_generate=$ELSES_ROOT/bin/elses-xml-generate

# make -j H2_construct

for nx in 12; do
    ny=1
    nz=1
    # Generate the xml file from the source geometry depenending on the number of repetitions specified.
    $exec_supercell $nx $ny $nz $source_file

    cp C60_fcc.xyz $ELSES_ROOT/make_supercell_C60_FCCs_w_noise

    # generate config.xml.
    $exec_elses_xml_generate $ELSES_ROOT/make_supercell_C60_FCCs_w_noise/generate.xml $fcc_xml_file

    # Calcualte dimension of the resulting matrix.
    N=$(($nx * $ny * $nz * 1 * 1 * 1 * 32 * 60 * 4))
    NLEAF=240
    MAX_RANK=100

    # Values from Ridwan's paper where the correct k-th eigen value of the matrix resides.
    interval_start=0
    interval_end=2048
    mpiexec bin/H2_construct --N $N \
           --ndim 3 \
           --nleaf $NLEAF \
           --max_rank $MAX_RANK \
           --kernel_func elses_c60 \
           --kind_of_geometry elses_c60_geometry \
           --admis_kind diagonal \
           --geometry_file C60_fcc.xyz \
           --param_1 $interval_start --param_2 $interval_end \
           --use_nested_basis 1
done
