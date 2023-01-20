#!/bin/bash
#$ -cwd
#$ -l rt_F=8
#$ -l h_rt=2:00:00
#$ -N HSS_P8
#$ -o HSS_P8_out.log
#$ -e HSS_P8_err.log

source ~/.bashrc

module purge
module load intel-mpi/2021.5 gcc intel-mkl/2022.0.0 cmake/3.22.3 intel-vtune intel-itac

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64/pkgconfig:/home/acb10922qh/gitrepos/googletest/build/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64

export OMP_PLACES=cores
# export VT_CONFIG=/home/acb10922qh/gitrepos/Hatrix/vt_config.conf

# make clean
make -j H2_dtd

# rm gmon.out-*
# export GMON_OUT_PREFIX=gmon.out-

for adm in 7; do
    ndim=2

    for nleaf in 512; do
        for max_rank in 25 50; do
    	    for N in 65536; do
                mpirun -n 8 -ppn 1 -f $SGE_JOB_HOSTLIST \
                       ./bin/H2_dtd --N $N \
               	       --nleaf $nleaf \
               	       --kernel_func laplace \
               	       --kind_of_geometry grid \
               	       --ndim $ndim \
               	       --max_rank $max_rank \
               	       --accuracy 1e-8 \
               	       --admis $adm \
               	       --admis_kind geometry \
               	       --construct_algorithm miro \
               	       --add_diag 1e-8 \
               	       --use_nested_basis
    	    done
        done
    done
done

# gprof -s bin/H2_dtd gmon.out-*
# gprof -q bin/H2_dtd gmon.sum > NEW.gprof
