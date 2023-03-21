#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig:/Users/sameer/gitrepos/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib
export PATH=/Users/sameer/gitrepos/parsec/build/bin:$PATH

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp
ROOT=$PWD

set -e

make clean

# export VECLIB_MAXIMUM_THREADS=1
cd examples/distributed/H2_ptg
./compile_jdf.sh
cd $ROOT

ulimit -c unlimited

make -j H2_ptg
# make -j H2_dtd

max_rank=50
ndim=1
nleaf=256
adm=0

for N in 1024; do
    # for adm in 1; do
    #     for nleaf in 512; do
    #         for max_rank in 200; do
    #             ./bin/H2_main --N $N \
    #                           --nleaf $nleaf \
    #                           --kernel_func gsl_matern \
    #                           --kind_of_geometry grid \
    #                           --ndim $ndim \
    #                           --max_rank $max_rank \
    #                           --accuracy -1 \
    #                           --admis $adm \
    #                           --admis_kind geometry \
    #                           --construct_algorithm miro \
    #                           --param_1 1e-2 --param_2 0.5 --param_3 0.1 \
    #                           --kind_of_recompression 3
    #         done
    #     done
    # done
    lldb -- ./bin/H2_ptg --N $N \
                 --nleaf $nleaf \
                 --kernel_func laplace \
                 --kind_of_geometry grid \
                 --ndim $ndim \
                 --max_rank $max_rank \
                 --accuracy -1 \
                 --admis $adm \
                 --admis_kind diagonal \
                 --construct_algorithm miro \
                 --param_1 1e-9  \
                 --kind_of_recompression 3
done

function benchmark_sc22() {
    mpicxx -I${VEC_LIB_INCLUDE} -I/opt/homebrew/opt/lapack/include -I/Users/sameer/gitrepos/gsl-2.7.1/build/include -framework Accelerate -L/Users/sameer/gitrepos/gsl-2.7.1/build/lib -lgsl -lm -L/opt/homebrew/opt/lapack/lib -llapacke -llapack examples/SymmH2_ULV_SC22.cpp -o bin/sc_22

    for N in 32768; do
        for nleaf in 512 1024 2048; do
            for max_rank in 150 200 500; do
                mpirun -n 8 bin/sc_22 $N 2 $nleaf 1.e-8 $max_rank 2000
            done
        done
    done
}

# benchmark_sc22
