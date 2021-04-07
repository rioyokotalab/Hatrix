TOPSRCDIR = ../..
include $(TOPSRCDIR)/make.inc

.PHONY: clean

.SUFFIXES: .cpp .cu
all: functions

FUNCTIONS = blas.o lapack.o

functions: $(FUNCTIONS)

clean:
	$(RM) -f $(FUNCTIONS)
