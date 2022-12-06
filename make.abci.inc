BLAS_INCLUDE = -m64  -I"${MKLROOT}/include" -DUSE_MKL
BLAS_LIB     =  ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a \
             -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
             ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a \
             ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -lgomp -lpthread -lm -ldl
CXXFLAGS     = -O0 -g3 -ggdb  -std=c++11 -fopenmp -I. -Wall -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE)
LDFLAGS      = $(shell pkg-config --libs gtest_main gtest) $(LAPACK_LIB) $(BLAS_LIB) -std=c++11 -lstdc++ -fopenmp
PARSEC_INCLUDE = $(shell pkg-config --cflags parsec)
PARSEC_LIB     = $(shell pkg-config --libs parsec)
SCALAPACK_LIB = $(BLAS_LIB)
LDFLAGS      = $(BLAS_LIB)
CXX          = g++
MPICXX       = mpicxx
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
