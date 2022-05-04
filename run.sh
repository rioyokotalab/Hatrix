#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J HSS
#SBATCH --time=72:00:00

source /etc/profile.d/modules.sh
module load cmake lapack/3.9.0 openmpi/4.0.5 gcc/7.5

# source ~/.bashrc
make clean
make -j

for N in 2048; do
    ./bin/HSS_main --N $N --nleaf 128 --kernel-func laplace --add-diag 1e-6 \
                   --rank 15 --nested-basis 1 --construct-algorithm miro \
                   --kind-of-geometry circular
done
