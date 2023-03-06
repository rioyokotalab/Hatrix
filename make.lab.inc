BLAS_INCLUDE = -DMKL_LP64 -m64 -I${MKLROOT}/include -DUSE_MKL
BLAS_LIB     = ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a \
             -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
             ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a \
             ${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lgomp \
             -lpthread -lm -ldl
PAPI_INCLUDE = $(shell pkg-config --cflags papi)
PAPI_LIBS    = $(shell pkg-config --libs papi)
CXXFLAGS     = -std=c++14 -O3 -g -fopenmp -I. -Wall -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE) \
               $(shell pkg-config --cflags gtest gtest_main) $(PAPI_INCLUDE) -DUSE_MKL
CXX          = g++
MPICXX       = mpicxx
LDFLAGS      = $(shell pkg-config --libs gtest_main gtest) $(BLAS_LIB) $(PAPI_LIBS) -lgfortran
PARSEC_INCLUDE = $(shell pkg-config --cflags parsec)
PARSEC_LIB     = $(shell pkg-config --libs parsec)
SCALAPACK_LIB = $(BLAS_LIB)
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
