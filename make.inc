BLAS_INCLUDE = -DMKL_ILP64  -m64  -I${MKLROOT}/include
BLAS_LIB = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
