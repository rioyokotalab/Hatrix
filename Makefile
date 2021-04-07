TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

.SUFFIXES: .cpp .cu

CXXFLAGS = $(BLAS_INCLUDE) -std=c++14 -g -rdynamic -O0 -fopenmp -I. -Wall

CXX = g++ ${CXXFLAGS}
LDFLAGS=  $(BLAS_LIB)

SOURCES =

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $?

gemm: test/gemm.o $(SOURCES)
	$(CXX) $(LDFLAGS) $? -o $@

clean:
	$(RM) *.o *.out *.xml main
