LAPACK_INCLUDE =
LAPACK_LIB     = -L$FJSVXTCLANGA/lib64 -lfjlapacksve
BLAS_INCLUDE   =
BLAS_LIB       = -SSL2
CXX            = FCC
MPICXX         = mpiFCC
CXXFLAGS       = -Nclang -O3 -fopenmp -g $(shell pkg-config --cflags gtest gtest_main) \
                 -I. -I$(TOPSRCDIR)/include/ -fPIC -Ofast -std=c++11 -stdlib=libstdc++
LDFLAGS        = $(shell pkg-config --libs gtest_main gtest) $(LAPACK_LIB) $(BLAS_LIB) -std=c++11 -lstdc++ -fopenmp
SCALAPACK_LIB  =
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@