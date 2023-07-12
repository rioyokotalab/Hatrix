BLAS_INCLUDE   = -DMKL_LP64 -I${MKLROOT}/include -DUSE_MKL
BLAS_LIB       = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 \
                 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl \
                 $(shell pkg-config --cflags gtest)
SCALAPACK_LIB  = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 \
                 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 \
                 -lpthread -lm -ldl
ELPA_INCLUDE   = -I/home/ridwan/local/include/elpa-2023.05.001
ELPA_LIB       = -L/home/ridwan/local/lib -lelpa
CXXFLAGS       = -std=c++11 -O3 -fopenmp -I. -Wall -I$(TOPSRCDIR)/include/ $(BLAS_INCLUDE) \
                 $(shell pkg-config --cflags gtest) $(shell pkg-config --cflags gtest_main) \
                 -Wno-maybe-uninitialized -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-compare \
                 -I$(TOPSRCDIR)/examples/common/
CXX            = g++
MPICXX         = mpicxx
LDFLAGS        = $(shell pkg-config --libs gtest_main) $(shell pkg-config --libs gtest)  $(BLAS_LIB)
RM             = rm -rf
AR             = ar
ARFLAGS        = -rv
MV             = mv -f

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@
