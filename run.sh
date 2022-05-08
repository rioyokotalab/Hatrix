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

echo "CONSTANT ACCURACY CONSTRUCTION"
for data in "1024 1e-5" "2048 1e-5" "4096 1e-6" "8192 1e-6" "16384 1e-7" "32768 1e-7" "65536 1e-8" "131072 1e-8" "262144 1e-9" "524288 1e-9" "1048576 1e-10"; do
    set -- $data
    for construction in id_random miro;  do
        ./bin/HSS_main --N $1 --nleaf 128 --kernel-func laplace --add-diag $2 \
                       --acc 1e-9 --nested-basis 1 --construct-algorithm $construction \
                       --kind-of-geometry circular
    done
done
