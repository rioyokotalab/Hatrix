#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J HSS
#SBATCH --time=72:00:00

source /etc/profile.d/modules.sh
module load cmake lapack/3.9.0 openmpi/4.0.5 gcc/7.5

# valgrind warnings https://stackoverflow.com/questions/36197527/insight-as-to-why-valgrind-shows-memory-leak-for-intels-mkl-lapacke
# export MKL_DISABLE_FAST_MM=1
# source ~/.bashrc
# make clean#
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
make -j

for N in 1024 2048 4096 8192 16384 32768 65536; do
    ./bin/HSS_main --N $N --nleaf 128 --kernel-func laplace --add-diag 1e-6 \
                   --rank 15 --nested-basis 1 --construct-algorithm miro \
                   --kind-of-geometry circular
done
