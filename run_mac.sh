#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig:/Users/sameer/gitrepos/gsl-2.7.1/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib
export PATH=/Users/sameer/gitrepos/parsec/build/bin:$PATH

# Keep for this reason: https://github.com/open-mpi/ompi/issues/7393
export TMPDIR=/tmp
ROOT=$PWD

set -e

# make clean

# export VECLIB_MAXIMUM_THREADS=1
# cd examples/distributed/H2_ptg
# ./compile_jdf.sh
# cd $ROOT

# ulimit -c unlimited

make -j H2_main
# make -j Dense

max_rank=50
nleaf=512
adm=0
ndim=1
N=4096

# ./bin/Dense --N $N --kernel_func laplace --kind_of_geometry grid --ndim $ndim --param_1 1e-9
# ./bin/Dense --N $N --kernel_func laplace --kind_of_geometry grid --ndim $ndim --param_1 1e-4
# ./bin/Dense --N $N --kernel_func gsl_matern --kind_of_geometry grid --ndim $ndim --param_1 1 --param_2 0.03 --param_3 0.5
# ./bin/Dense --N $N --kernel_func gsl_matern --kind_of_geometry grid --ndim $ndim --param_1 1 --param_2 0.1 --param_3 1
# ./bin/Dense --N $N --kernel_func yukawa --kind_of_geometry grid --ndim $ndim --param_1 1e-9 --param_2 1
# ./bin/Dense --N $N --kernel_func yukawa --kind_of_geometry grid --ndim $ndim --param_1 1e-4 --param_2 1

for N in 64; do
    for adm in 1; do
        for nleaf in 32; do
            for max_rank in 20; do
                ./bin/H2_main --N $N \
                              --nleaf $nleaf \
                              --kernel_func gsl_matern \
                              --kind_of_geometry grid \
                              --ndim $ndim \
                              --max_rank $max_rank \
                              --accuracy -1 \
                              --admis $adm \
                              --admis_kind geometry \
                              --construct_algorithm miro \
                              --param_1 1 --param_2 0.03 --param_3 0.5 \
                              --kind_of_recompression 3
            done
        done
    done
    # lldb -- ./bin/H2_ptg --N $N \
    #              --nleaf $nleaf \
    #              --kernel_func laplace \
    #              --kind_of_geometry grid \
    #              --ndim $ndim \
    #              --max_rank $max_rank \
    #              --accuracy -1 \
    #              --admis $adm \
    #              --admis_kind diagonal \
    #              --construct_algorithm miro \
    #              --param_1 1e-9  \
    #              --kind_of_recompression 3
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
