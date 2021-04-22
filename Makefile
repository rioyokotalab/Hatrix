TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

DIRS := src/classes src/functions
OBJLIBS := libclasses.a libfunctions.a
TEST := test
EXECUTABLES := matmul lu qr block_dense_lu Matrix

.PHONY: dirs $(DIRS)
dirs: $(DIRS)

all: $(EXECUTABLES)

$(DIRS):
	$(MAKE) -C $@

LINK_EXECUTABLE = $(CXX) $< $(OBJLIBS) $(LDFLAGS)  -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

matmul: $(TEST)/matmul.o dirs
	$(LINK_EXECUTABLE)

lu: $(TEST)/lu.o dirs
	$(LINK_EXECUTABLE)

qr: $(TEST)/qr.o dirs
	$(LINK_EXECUTABLE)

Matrix: $(TEST)/Matrix.o dirs
	$(LINK_EXECUTABLE)

block_dense_lu: $(TEST)/block_dense_lu.o dirs
	$(LINK_EXECUTABLE)

.PHONY: clean
clean:
	for dir in $(DIRS) test; do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) bin/
