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
make -j HSS_main

echo "CONSTANT ACCURACY CONSTRUCTION"
for data in "1024 1e-5"; do
    set -- $data
    ./bin/HSS_main --N $1 --nleaf 128 --kernel-func laplace --add-diag $2 \
                   --acc 1e-9 --nested-basis 1 --construct-algorithm id_random \
                   --kind-of-geometry circular

    # ./bin/HSS_main --N $1 --nleaf 128 --kernel-func laplace --add-diag $2 \
    #                --acc 1e-9 --nested-basis 1 --construct-algorithm miro \
    #                --kind-of-geometry circular
done
