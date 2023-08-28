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

make -j H2_construct

for N in 2048 4096 8192 16384 32768 131072 262144 524288 1048576; do
    nx=1
    ny=1
    nz=1
# Generate the xml file from the source geometry depenending on the number of repetitions specified.
    $exec_supercell $nx $ny $nz $source_file

    cp C60_fcc.xyz $ELSES_ROOT/make_supercell_C60_FCCs_w_noise

    # generate config.xml.
    $exec_elses_xml_generate \
        $ELSES_ROOT/make_supercell_C60_FCCs_w_noise/generate.xml \
        $fcc_xml_file

    # Calcualte dimension of the resulting matrix.
    # N=$(($nx * $ny * $nz * 1 * 1 * 1 * 32 * 60 * 4))
    # N=2048
    for MAX_RANK in 30; do
        NLEAF=128
        NDIM=2
        KERNEL_FUNC=laplace

        # Values from Ridwan's paper where the correct k-th eigen value of the matrix resides.
        # interval_start=0
        # interval_end=2048

        # Laplace kernel paramters
        p1=1e-9
        mpiexec bin/H2_construct --N $N \
               --ndim $NDIM \
               --nleaf $NLEAF \
               --max_rank $MAX_RANK \
               --kernel_func $KERNEL_FUNC \
               --kind_of_geometry grid \
               --admis_kind geometry \
               --admis 0.7 \
               --geometry_file C60_fcc.xyz \
               --param_1 $p1 \
               --use_nested_basis 0
    done
done
