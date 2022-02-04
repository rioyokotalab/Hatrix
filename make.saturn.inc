BLAS_INCLUDE = -DMKL_LP64 -I${MKLROOT}/include -DUSE_MKL
BLAS_LIB     = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 \
               -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
CXX          = g++ -g -std=c++11
CXXFLAGS     = -std=c++11 -rdynamic -O0 -fopenmp -I. -Wall -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE) \
               $(shell pkg-config --cflags gtest_main gtest starsh)
LDFLAGS      = $(shell pkg-config --libs gtest_main gtest starsh)  $(BLAS_LIB)
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

.SUFFIXES: .cpp
