Various implementations of HSS and H2 matrices.

# HSS

Make commmand:
```
make HSS_main
```

Features:
1. HSS matrix construction using ID or QR.
2. Matvec.
3. Shared memory using vendor BLAS.
4. Supports diagonal admis conditions.
5. Works with circular, grid geometry.
6. Work with laplace kernel.

Sample command for QR based factorization:
```
./bin/HSS_main --N 8192 \
           --nleaf 512 \
           --kernel_func laplace \
           --kind_of_geometry circular \
           --ndim 1 \
           --max_rank 100 \
           --accuracy 1e-11 \
           --admis 0 \
           --admis_kind diagonal \
           --construct_algorithm miro \
           --add_diag 1e-5 \
           --use_nested_basis
```
