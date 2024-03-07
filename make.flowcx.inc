BLAS_INCLUDE = -DMKL_LP64 -I${MKLROOT}/include -DUSE_MKL
BLAS_LIB     = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 \
               -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
PARSEC_INCLUDE = $(shell pkg-config --cflags parsec)
PARSEC_LIB     = $(shell pkg-config --libs parsec)
LDFLAGS        = $(BLAS_LIB) -L/home/z44294z/gitrepos/gsl-2.7.1/build/lib -lgsl -lm -fopenmp
CXXFLAGS     = -std=c++11 -O3 -fopenmp -I. -Wall -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE) \
             $(shell pkg-config --cflags gsl)
ELSES_OBJ    = ${ELSES_ROOT}/src/src.a ${ELSES_ROOT}/xmlf90-1.2g-elses/macros/lib/libflib.a
CXX          = g++
MPICXX       = mpicxx
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
