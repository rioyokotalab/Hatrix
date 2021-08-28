BLAS_INCLUDE = -DMKL_LP64 -I${MKLROOT}/include
BLAS_LIB     = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 \
               -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl \
               $(shell pkg-config --cflags gtest)
CXXFLAGS     = -std=c++14 -rdynamic -O3 -fopenmp -I. -Wall -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE) \
               $(shell pkg-config --cflags gtest) $(shell pkg-config --cflags gtest_main)
CXX          = g++ -g
LDFLAGS      = $(shell pkg-config --libs gtest_main) $(shell pkg-config --libs gtest)  $(BLAS_LIB)
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

.SUFFIXES: .cpp
