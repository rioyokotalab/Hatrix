BLAS_INCLUDE =  -m64  -I"${MKLROOT}/include" -DUSE_MKL
BLAS_LIB     =  -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -Wl,--no-as-needed \
             -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp -lpthread -lm -ldl
PAPI_INCLUDE = $(shell pkg-config --cflags papi)
PAPI_LIBS    = $(shell pkg-config --libs papi)
CXXFLAGS     = -std=c++17 -O3 -g -fopenmp -I. -Wall -I$(TOPSRCDIR)/include/ \
               $(BLAS_INCLUDE) -I/home/sameer.deshmukh/gsl-2.7.1/build/include \
               $(shell pkg-config --cflags gtest gtest_main) $(PAPI_INCLUDE) -DUSE_MKL
CXX          = g++
MPICXX       = mpicxx
LDFLAGS      = $(BLAS_LIB) $(shell pkg-config --libs gtest_main gtest) $(PAPI_LIBS) -lgfortran \
               -L/home/sameer.deshmukh/gsl-2.7.1/build/lib -lgsl
PARSEC_INCLUDE = $(shell pkg-config --cflags parsec)
PARSEC_LIB     = $(shell pkg-config --libs parsec)
SCALAPACK_LIB = $(BLAS_LIB)
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
