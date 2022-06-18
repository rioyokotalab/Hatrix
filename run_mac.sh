#!/bin/bash

make all
export TMPDIR=/tmp

for N in 8192; do
    for matrix_type in 1; do
        for admis in 1000; do
            for rank in 20; do
                echo "begin exec"
                ./bin/UMV_H2_far_dense $N $rank 128 $admis 3 geometry_admis 0 $matrix_type
            done
        done
    done
done

# for nprocs in 1 2 4 8; do
#     mpirun -n $nprocs ./bin/HSS_scalapack --N 16384 \
#            --nleaf 1024 \
#            --kernel_func laplace \
#            --kind_of_geometry circular \
#            --ndim 1 \
#            --max_rank 100 \
#            --accuracy 1e-11 \
#            --admis 0 \
#            --admis_kind diagonal \
#            --construct_algorithm miro \
#            --add_diag 1e-5 \
#            --use_nested_basis
# done
