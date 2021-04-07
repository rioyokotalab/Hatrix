TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

.SUFFIXES: .cpp .cu
DIRS := src/classes src/functions test
GEMM := gemm
OBJLIBS := src/classes/Matrix.o src/functions/blas.o src/functions/lapack.o \
test/gemm.o
TEST_DIR := test

.PHONY: all $(GEMM) $(DIRS)

all: $(GEMM)

$(GEMM) $(DIRS):
	$(MAKE) --directory=$@

$(GEMM): $(DIRS)
	$(CXX) $(LDFLAGS) $(OBJLIBS)  -o $(GEMM) 

clean: $(GEMM) $(DIRS)
	$(MAKE) --directory=$@ clean

