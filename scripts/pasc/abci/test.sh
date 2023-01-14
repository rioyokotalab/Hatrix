#!/bin/bash
#$ -cwd
#$ -l rt_F=16
#$ -l h_rt=2:00:00
#$ -N NEW
#$ -o NEW_out.log
#$ -e NEW_err.log

source ~/.bashrc

module purge
module load intel-mpi/2021.5 gcc intel-mkl/2022.0.0 cmake/3.22.3 intel-vtune intel-itac

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64/pkgconfig:/home/acb10922qh/gitrepos/googletest/build/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64

export OMP_PLACES=cores

make clean
make -j H2_dtd
# mpirun -n $procs -ppn 2 -f $SGE_JOB_HOSTLIST
                   # -gtool "vtune -collect hpc-performance -r vtune-mpi:all" \
                       # -trace -trace-pt2pt \
rm gmon.out-*
export GMON_OUT_PREFIX=gmon.out-

for adm in 0.8; do
    ndim=3
    max_rank=50
    for nleaf in 1024 ; do
    	for N in 131072; do
            mpirun -n 16 -ppn 1 -f $SGE_JOB_HOSTLIST -l \
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

gprof -s bin/H2_dtd gmon.out-*
gprof -q bin/H2_dtd gmon.sum > NEW.gprof
