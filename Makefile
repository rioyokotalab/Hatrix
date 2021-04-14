TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

DIRS := src/classes src/functions
OBJLIBS := libclasses.a libfunctions.a
TEST := test
EXECUTABLES := gemm getrf qr block_dense_lu

.PHONY: dirs $(DIRS)
dirs: $(DIRS)

all: $(EXECUTABLES)

$(DIRS):
	$(MAKE) -C $@

gemm: test/gemm.o dirs
	$(CXX) test/$@.o $(OBJLIBS) $(LDFLAGS)  -o $@

getrf: test/getrf.o dirs
	$(CXX) $(TEST)/$@.o $(OBJLIBS) $(LDFLAGS)  -o $@

qr: $(TEST)/qr.o dirs
	$(CXX) $(TEST)/$@.o $(OBJLIBS) $(LDFLAGS)  -o $@

block_dense_lu: $(TEST)/block_dense_lu.o dirs
	$(CXX) $(TEST)/$@.o $(OBJLIBS) $(LDFLAGS)  -o $@

.PHONY: clean
clean:
	for dir in $(DIRS) test; do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) $(EXECUTABLES)
