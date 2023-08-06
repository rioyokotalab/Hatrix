BLAS_INCLUDE   =
BLAS_LIB       = -SSL2
LAPACK_INCLUDE =
LAPACK_LIB     = -L$FJSVXTCLANGA/lib64 -lfjlapacksve
SCALAPACK_LIB  =
CXX            = FCC
MPICXX         = mpiFCC
CXXFLAGS       = -Nclang -O3 -fopenmp $(shell pkg-config --cflags gtest gtest_main) \
                 -I. -I$(TOPSRCDIR)/include/ -I$(TOPSRCDIR)/examples/common \
                 -fPIC -Ofast -std=c++11 -stdlib=libstdc++ -Kfast,openmp \
                 -I/home/hp190122/u01594/gsl-2.7.1/build/include
LDFLAGS        = $(shell pkg-config --libs gtest_main gtest) $(LAPACK_LIB) $(BLAS_LIB) \
                 -std=c++11 -lstdc++ -fopenmp \
                 -L/home/hp190122/u01594/gsl-2.7.1/build/lib -lgsl
PARSEC_INCLUDE = $(shell pkg-config --cflags parsec)
PARSEC_LIB     = $(shell pkg-config --libs parsec)
SCALAPACK_LIB  = -SCALAPACK -SSL2BLAMP
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f
ELSES_OBJ    = ${ELSES_ROOT}/src/src.a ${ELSES_ROOT}/xmlf90-1.2g-elses/macros/lib/libflib.a

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
