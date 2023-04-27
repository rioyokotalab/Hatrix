BLAS_INCLUDE   = $(shell pkg-config --cflags cblas blas)
BLAS_LIB       = $(shell pkg-config --libs cblas blas)
LAPACK_INCLUDE = $(shell pkg-config --cflags lapack lapacke)
LAPACK_LIB     = $(shell pkg-config --libs lapack lapacke)
SCALAPACK_LIB  =
CXX            = g++
MPICXX         = mpicxx
CXXFLAGS       = -std=c++11 -rdynamic -O3 -fopenmp -I. -Wall -I$(TOPSRCDIR)/include/ \
                 $(shell pkg-config --cflags gtest gtest_main) \
                 $(BLAS_INCLUDE) $(LAPACK_INCLUDE)
LDFLAGS        = $(shell pkg-config --libs gtest_main gtest) $(BLAS_LIB) $(LAPACK_LIB) -lgfortran
RM             = rm -rf
AR             = ar
ARFLAGS        = -rv
MV             = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
