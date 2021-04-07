TOPSRCDIR = ..
include $(TOPSRCDIR)/make.inc

.PHONY: clean

.SUFFIXES: .cpp .cu
all: functions

TEST = gemm.o

functions: $(TEST)

clean:
	$(RM) -f $(TEST)
