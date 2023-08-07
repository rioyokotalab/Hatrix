#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J H2_gsl
#SBATCH --time=24:00:00

set -e

source ~/.bashrc
# source /home/sameer.deshmukh/gitrepos/parsec/build/bin/parsec.env.sh

source /etc/profile.d/modules.sh
module purge
module load gcc/12.2 cuda intel/2022/mkl cmake intel/2022/mpi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gsl-2.7.1/build/lib:/home/sameer.deshmukh/gitrepos/parsec/build/lib

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64:/home/sameer.deshmukh/gitrepos/papi/src/lib:/home/sameer.deshmukh/gitrepos/gsl-2.7.1/build/lib

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=32

# Generate the points for the ELSES matrix.
export ELSES_ROOT=/home/sameer.deshmukh/ELSES_mat_calc-master
mol_folder=$ELSES_ROOT/sample/sample_non_geno/C60_fcc2x2x2_disorder_expand_1x1x1
source_file=$mol_folder/C60_fcc2x2x2_20220727.xyz
fcc_xml_file=C60_fcc.xml

exec_supercell=$ELSES_ROOT/make_supercell_C60_FCCs_w_noise/a.out
exec_elses_xml_generate=$ELSES_ROOT/bin/elses-xml-generate

make -j H2_construct

for nx in 1; do
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
    N=32
    NLEAF=8
    MAX_RANK=4
    NDIM=1
    KERNEL_FUNC=laplace

    # Values from Ridwan's paper where the correct k-th eigen value of the matrix resides.
    interval_start=0
    interval_end=2048
    mpirun -n 1 gdb -ex run --args ./bin/H2_construct --N $N \
           --ndim $NDIM \
           --nleaf $NLEAF \
           --max_rank $MAX_RANK \
           --kernel_func $KERNEL_FUNC \
           --kind_of_geometry grid \
           --admis_kind geometry \
           --admis 0.3 \
           --geometry_file C60_fcc.xyz \
           --param_1 $interval_start --param_2 $interval_end \
           --use_nested_basis 1
done
