BLAS_INCLUDE = -DMKL_LP64 -I${MKLROOT}/include -DUSE_MKL
BLAS_LIB     = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 \
               -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
CXXFLAGS     = -std=c++11 -O3 -fopenmp -I. -Wall -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE)
LDFLAGS      = $(BLAS_LIB)
CXX          = g++
MPICXX       = mpicxx
RM           = rm -rf
AR           = ar
ARFLAGS      = -rv
MV           = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
