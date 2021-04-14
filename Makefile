TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

DIRS := src/classes src/functions
OBJLIBS := libclasses.a libfunctions.a

.PHONY: dirs $(DIRS)
dirs: $(DIRS)

$(DIRS):
	$(MAKE) -C $@

gemm: test/gemm.o $(DIRS)
	$(CXX) test/gemm.o $(OBJLIBS) $(LDFLAGS)  -o $@

.PHONY: clean
clean:
	for dir in $(DIRS) test; do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) gemm
