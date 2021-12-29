BLAS_INCLUDE = -I${VEC_LIB_INCLUDE}
BLAS_LIB     = -framework Accelerate
CXX          = clang++ -std=c++11
CXXFLAGS     = -O3 -I. -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE) \
             $(shell pkg-config --cflags gtest gtest_main)
LDFLAGS      = $(shell pkg-config --libs gtest_main) $(shell pkg-config --libs gtest)  $(BLAS_LIB)
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

.SUFFIXES: .cpp
