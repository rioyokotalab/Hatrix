TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

DIRS := src/classes src/functions test
OBJLIBS := lib_classes.a lib_functions.a

.PHONY: dirs $(DIRS)
dirs: $(DIRS)

$(DIRS):
	$(MAKE) -C $@

gemm: test/gemm.o $(DIRS)
	$(CXX) $(LDFLAGS) $(OBJLIBS)  -o $@

.PHONY: clean
clean:
	for dir in $(DIRS); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
