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

gemm: $(TEST)/gemm.o dirs
	$(CXX) test/$@.o $(OBJLIBS) $(LDFLAGS)  -o $@
	mkdir -p bin
	$(MV) $@ bin/

getrf: $(TEST)/getrf.o dirs
	$(CXX) $(TEST)/$@.o $(OBJLIBS) $(LDFLAGS)  -o $@
	mkdir -p bin
	$(MV) $@ bin/

qr: $(TEST)/qr.o dirs
	$(CXX) $(TEST)/$@.o $(OBJLIBS) $(LDFLAGS)  -o $@
	mkdir -p bin
	$(MV) $@ bin/

block_dense_lu: $(TEST)/block_dense_lu.o dirs
	$(CXX) $(TEST)/$@.o $(OBJLIBS) $(LDFLAGS)  -o $@
	mkdir -p bin
	$(MV) $@ bin/

.PHONY: clean
clean:
	for dir in $(DIRS) test; do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) bin/
