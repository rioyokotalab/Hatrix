#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J BLR2_construct
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

mkdir build

pushd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j VERBOSE=1
popd

for N in 512 2048; do
    ./build/examples/H2_strong_CON_sameer \
        $N 64 0 40 0.5 0 1 3 1
done
