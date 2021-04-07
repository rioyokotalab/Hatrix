TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

.SUFFIXES: .cpp .cu
CLASSES_DIR := src/classes
FUNCTIONS_DIR := src/functions
TEST_DIR := test

MODULES := $(CLASSES_DIR) $(FUNCTIONS_DIR)

CXXFLAGS = -Iinclude/ $(BLAS_INCLUDE) -std=c++11 -g -rdynamic -O0 \
	-fopenmp -I. -Wall

CXX = g++ ${CXXFLAGS}
LDFLAGS=  $(BLAS_LIB)

SOURCES = src/classes/Matrix.o src/functions/blas.o src/functions/lapack.o

.PHONY: all  $(MODULES) $(TEST_DIR)
all: $(TEST_DIR)

$(TEST_DIR) $(MODULES):
	$(MAKE) --directory=$@

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $?

gemm: test/gemm.o $(SOURCES)
	$(CXX) $(LDFLAGS) gemm.o Matrix.o blas.o lapack.o -o $@

clean:
	$(RM) *.o *.out *.xml main
