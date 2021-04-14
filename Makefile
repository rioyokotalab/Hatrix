TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

DIRS := src/classes src/functions test
OBJLIBS := src/classes/libclasses.a src/functions/libfunctions.a

.PHONY: dirs $(DIRS)
dirs: $(DIRS)

$(DIRS):
	$(MAKE) -C $@

gemm: test/gemm.o $(DIRS)
	$(CXX)  test/gemm.o $(OBJLIBS) $(LDFLAGS)  -o $@

.PHONY: clean
clean:
	for dir in $(DIRS); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
