TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

DIRS := src/classes src/functions test

dirs: $(DIRS)

$(DIRS):
	$(MAKE) -C $@

gemm: $(DIRS)
	$(CXX) $(LDFLAGS) $(OBJLIBS)  -o $@

.PHONY: clean
clean:
	for dir in $(DIRS); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
