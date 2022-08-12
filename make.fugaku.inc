LAPACK_INCLUDE =
LAPACK_LIB     =
BLAS_INCLUDE   =
BLAS_LIB       =
CXX            = FCC
MPICXX         = mpiFCC
CXXFLAGS       = -Nclang -O3 -g --std=c++11
LDFLAGS        =
SCALAPACK_LIB  =
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
