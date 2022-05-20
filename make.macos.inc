LAPACK_INCLUDE = -I/opt/homebrew/opt/lapack/include
LAPACK_LIB = -L/opt/homebrew/opt/lapack/lib -llapacke -llapack
BLAS_INCLUDE = -I${VEC_LIB_INCLUDE}
BLAS_LIB     = -framework Accelerate
CXX          = clang++
MPICXX       = mpicxx
CXXFLAGS     = -O3 --std=c++11 -Xpreprocessor -fopenmp \
               -I"$(shell brew --prefix libomp)/include" \
               -I. -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE) $(LAPACK_INCLUDE) \
               $(shell pkg-config --cflags gtest gtest_main)
LDFLAGS      = $(shell pkg-config --libs gtest_main) $(shell pkg-config --libs gtest) \
               $(LAPACK_LIB) $(BLAS_LIB) -L"$(shell brew --prefix libomp)/lib" -lomp
SLATE_LIB    = -L/Users/sameer/gitrepos/slate/build/lib -lslate -llapackpp -lblaspp \
	       -rpath /Users/sameer/gitrepos/slate/build/lib
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
