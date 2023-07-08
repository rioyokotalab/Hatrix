#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J H2_gsl
#SBATCH --time=24:00:00

source ~/.bashrc
# source /home/sameer.deshmukh/gitrepos/parsec/build/bin/parsec.env.sh

source /etc/profile.d/modules.sh
module purge
module load gcc/12.2 cuda intel/2022/mkl cmake intel/2022/mpi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gsl-2.7.1/build/lib

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64:/home/sameer.deshmukh/gitrepos/papi/src/lib:/home/sameer.deshmukh/gitrepos/gsl-2.7.1/build/lib

export OMP_PLACES=cores
export OMP_PROC_BIND=close

ELSES_ROOT=/home/sameer.deshmukh/ELSES_mat_calc-master

exec_supercell=$ELSES_ROOT/make_supercell_C60_FCCs_w_noise/a.out
exec_elses_xml_generate=$ELSES_ROOT/bin/elses-xml-generate

make -j H2_eigen

# Generate the points for the ELSES matrix.
nx=1
ny=1
nz=1
source_file=$ELSES_ROOT/sample/sample_non_geno/C60_fcc2x2x2_disorder_expand_1x1x1/C60_fcc2x2x2_20220727.xyz

# Generate the geometry file.
$exec_supercell $nx $ny $nz $source_file

# generate config.xml.
$exec_elses_xml_generate $ELSES_ROOT/make_supercell_C60_FCCs_w_noise/generate.xml $ELSES_ROOT/sample/sample_non_geno/C60_fcc2x2x2_disorder_expand_1x1x1/C60_fcc2x2x2_20220727.xml

# copy config file into Hatrix root
cp $ELSES_ROOT/sample/sample_non_geno/C60_fcc2x2x2_disorder_expand_1x1x1/C60_fcc2x2x2_20220727.xml .
cp $ELSES_ROOT/sample/sample_non_geno/C60_fcc2x2x2_disorder_expand_1x1x1/config.xml .


./bin/H2_eigen --N 7680 \
               --nleaf 512 \
               --kernel_func elses_c60 \
               --kind_of_geometry elses_c60_geometry \
               --ndim 3 \
               --admis_kind geometry \
               --geometry_file C60_fcc.xyz \
               --use_nested_basis 1

rm *xml
rm *xyz
