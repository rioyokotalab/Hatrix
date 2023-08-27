#!/bin/bash
#PJM -L "node=16"
#PJM -L "rscgrp=cx-middle"
#PJM -L "elapse=12:00:00"
#PJM -e error.log
#PJM -j

module purge
module load gcc/8.4.0 openmpi cmake/3.25.2

source /home/center/opt/x86_64/cores/intel/compilers_and_libraries_2020.4.304/linux/mkl/bin/mklvars.sh intel64

PARSEC_PATH=/home/z44294z/gitrepos/parsec/install

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$PARSEC_PATH/lib64/pkgconfig:/home/z44294z/gitrepos/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PARSEC_PATH/lib64:/home/z44294z/gitrepos/gsl-2.7.1/build/lib:$MKLROOT/lib/intel64

export I_MPI_CXX=g++

export ELSES_ROOT=/home/z44294z/gitrepos/elses
mol_folder=$ELSES_ROOT/sample/sample_non_geno/C60_fcc2x2x2_disorder_expand_1x1x1
source_file=$mol_folder/C60_fcc2x2x2_20220727.xyz
fcc_xml_file=C60_fcc.xml

exec_supercell=$ELSES_ROOT/make_supercell_C60_FCCs_w_noise/a.out
exec_elses_xml_generate=$ELSES_ROOT/bin/elses-xml-generate

# make clean

ROOT=$PWD
cd examples/distributed/H2_construct
export MPICC=mpicc

$PARSEC_PATH/bin/parsec-ptgpp -E -i h2_factorize_flows.jdf -o h2_factorize_flows
$MPICC $(pkg-config --cflags parsec) -I../include/distributed -O0 -g \
       h2_factorize_flows.c -c -o h2_factorize_flows.o
cd $ROOT


make -j H2_construct

export MKL_NUM_THREADS=40

for N in 262144 1048576; do
    nx=1
    ny=1
    nz=1
    # Generate the xml file from the source geometry.
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
        NDIM=1
        KERNEL_FUNC=laplace
        ADMIS=0.4

        # Values from Ridwan's paper where the correct k-th eigen value of the matrix resides.
        # interval_start=0
        # interval_end=2048

        # --report-bindings

        # Laplace kernel paramters
        p1=1e-9
        mpirun -n $PJM_MPI_PROC -machinefile $PJM_O_NODEINF -npernode 1 \
               --report-bindings ./bin/H2_construct --N $N \
               --ndim $NDIM \
               --nleaf $NLEAF \
               --max_rank $MAX_RANK \
               --kernel_func $KERNEL_FUNC \
               --kind_of_geometry grid \
               --admis_kind geometry \
               --admis $ADMIS \
               --geometry_file C60_fcc.xyz \
               --param_1 $p1 \
               --use_nested_basis 0
    done
done
