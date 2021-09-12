BLAS_INCLUDE = $(shell pkg-config --cflags cblas lapacke)
BLAS_LIB     = $(shell pkg-config --libs lapacke lapack cblas blas)
CXXFLAGS     = -std=c++11 -rdynamic -O3 -fopenmp -I. -Wall -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE) \
               $(shell pkg-config --cflags gtest gtest_main)
CXX          = g++
LDFLAGS      = $(shell pkg-config --libs gtest_main gtest) $(BLAS_LIB) -lgfortran
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

.SUFFIXES: .cpp
