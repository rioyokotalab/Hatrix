LAPACK_INCLUDE = -I/opt/homebrew/opt/lapack/include
LAPACK_LIB = -L/opt/homebrew/opt/lapack/lib -llapacke -llapack
BLAS_INCLUDE = -I${VEC_LIB_INCLUDE}
BLAS_LIB     = -framework Accelerate
CXX          = clang++ --std=c++11 -Xpreprocessor -fopenmp \
             -I"$(shell brew --prefix libomp)/include"
CXXFLAGS     = -O3 -g -I. -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE) $(LAPACK_INCLUDE) \
             $(shell pkg-config --cflags gtest gtest_main)
LDFLAGS      = $(shell pkg-config --libs gtest_main) $(shell pkg-config --libs gtest) \
             $(LAPACK_LIB) $(BLAS_LIB) -L"$(shell brew --prefix libomp)/lib" -lomp
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

.SUFFIXES: .cpp
