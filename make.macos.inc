LAPACK_INCLUDE = -I/opt/homebrew/opt/lapack/include
LAPACK_LIB     = -L/opt/homebrew/opt/lapack/lib -llapacke -llapack
BLAS_INCLUDE   = -I${VEC_LIB_INCLUDE}
BLAS_LIB       = -framework Accelerate
CXX            = clang++
CC             = clang
MPICXX         = mpicxx
MPICC          = mpicc
CFLAGS         = -O0 -g -Wall -Xpreprocessor
CXXFLAGS       = -O0 -g -Wall --std=c++14 -Xpreprocessor \
                 -I. -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE) $(LAPACK_INCLUDE) \
                 $(shell pkg-config --cflags gtest gtest_main) \
                 -I"$(shell brew --prefix libomp)/include" \
                 -I/Users/sameerdeshmukh/gitrepos/gsl-2.7.1/build/include
LDFLAGS        = $(shell pkg-config --libs gtest_main gtest) \
                 $(LAPACK_LIB) $(BLAS_LIB) -L/Users/sameerdeshmukh/gitrepos/gsl-2.7.1/build/lib -lgsl -lm \
                 -L"$(shell brew --prefix libomp)/lib" -lomp
SLATE_LIB      = -L/Users/sameerdeshmukh/gitrepos/slate/build/lib -lslate -llapackpp -lblaspp \
	         -rpath /Users/sameerdeshmukh/gitrepos/slate/build/lib
SCALAPACK_LIB   = -L/opt/homebrew/Cellar/scalapack/2.2.0_1/lib -lscalapack
PARSEC_INCLUDE = $(shell pkg-config --cflags parsec)
PARSEC_LIB     = $(shell pkg-config --libs parsec) \
               -rpath /Users/sameerdeshmukh/gitrepos/parsec/install/lib
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
