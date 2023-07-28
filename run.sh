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

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gsl-2.7.1/build/lib:/home/sameer.deshmukh/gitrepos/parsec/build/lib

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/papi/src/lib/pkgconfig:/home/sameer.deshmukh/gitrepos/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sameer.deshmukh/gitrepos/parsec/build/lib:/mnt/nfs/packages/x86_64/cuda/cuda-11.7/lib64:/home/sameer.deshmukh/gitrepos/papi/src/lib:/home/sameer.deshmukh/gitrepos/gsl-2.7.1/build/lib

export OMP_PLACES=cores
export OMP_PROC_BIND=close

N=7680

# Matrix size 7,680

function generate_elses_config_file {
    ELSES_ROOT=/home/sameer.deshmukh/ELSES_mat_calc-master
    exec_supercell=$ELSES_ROOT/make_supercell_C60_FCCs_w_noise/a.out
    exec_elses_xml_generate=$ELSES_ROOT/bin/elses-xml-generate

    if [ $N == 7680 ]; then
        source elses_7680.sh
    elif [ $N == 30720 ]; then
        source elses_30720.sh
    fi

    # Generate the geometry file.
    $exec_supercell $nx $ny $nz $source_file

    # generate config.xml.
    $exec_elses_xml_generate $ELSES_ROOT/make_supercell_C60_FCCs_w_noise/generate.xml $ELSES_ROOT/sample/sample_non_geno/C60_fcc2x2x2_disorder_expand_1x1x1/C60_fcc2x2x2_20220727.xml

    # copy config file into Hatrix root
    cp $fcc_xml_file .
    cp $xml_config_file .
}

generate_elses_config_file

make -j H2_eigen

# values from Ridwan's paper where the correct k-th eigen value of the matrix resides.
interval_start=0
interval_end=2048

./bin/H2_eigen --N $N \
               --nleaf 240 \
               --kernel_func elses_c60 \
               --kind_of_geometry elses_c60_geometry \
               --admis_kind geometry \
               --geometry_file C60_fcc.xyz \
               --param_1 $interval_start --param_2 $interval_end \
               --use_nested_basis 1
