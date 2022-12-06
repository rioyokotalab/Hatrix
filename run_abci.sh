#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -N HATRIX
#$ -o HATRIX_out.log
#$ -e HATRIX_err.log

source ~/.bashrc

module purge
module load intel-mpi/2021.5 gcc/11.2.0 intel-mkl/2022.0.0 cmake/3.22.3

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64/pkgconfig:/home/acb10922qh/gitrepos/googletest/build/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/acb10922qh/gitrepos/parsec/build/lib64

export OMP_PLACES=cores

make -j H2_dtd


for adm in 0.8; do
    nleaf=512
    ndim=2
    max_rank=110

                        # mpirun -n 4 ./bin/H2_dtd --N $N \
    for N in 4096; do
	echo "running"
        # mpirun -n 4 -genv I_MPI_DEBUG=10  xterm -e gdb -ex=run --args ./bin/H2_dtd --N $N \
            mpirun -n 4 ./bin/H2_dtd --N $N \
               --nleaf $nleaf \
               --kernel_func laplace \
               --kind_of_geometry grid \
               --ndim $ndim \
               --max_rank $max_rank \
               --accuracy 1e-8 \
               --admis $adm \
               --admis_kind geometry \
               --construct_algorithm miro \
               --add_diag 1e-10 \
               --use_nested_basis
    done
done
