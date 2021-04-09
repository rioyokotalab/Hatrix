TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

DIRS := src/classes src/functions test
# OBJLIBS := $(DIRS:=.o)

# .PHONY: gemm dirs $(DIRS)
# all: gemm

dirs: $(DIRS)

$(DIRS):
	$(MAKE) -C $@

gemm: $(DIRS)
	$(CXX) $(LDFLAGS) $(OBJLIBS)  -o $@

.PHONY: clean
clean: $(DIRS)
	$(MAKE) -C $@
	rm *.a *.o gemm

