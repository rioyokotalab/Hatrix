#!/bin/bash

make HSS_main

for N in 1024; do
    for r in 30; do
        ./bin/HSS_main -v -h
        ./bin/HSS_main --N $N \
             --nleaf 128 \
             --kernel_func laplace \
             --kind_of_geometry circular \
             --ndim 1 \
             --max_rank $r \
             --accuracy 1e-11 \
             --admis 0 \
             --admis_kind diagonal \
             --construct_algorithm miro \
             --add_diag 1e-5 \
             --use_nested_basis \
             -v
    done
done

make HSS_slate
