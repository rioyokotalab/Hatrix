TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

DIRS := src/classes src/functions src/util
OBJLIBS := libclasses.a libfunctions.a libutil.a
TEST := test
EXECUTABLES := matmul lu qr block_dense_lu Matrix HSS_2level_construct

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

HSS_2level_construct: $(TEST)/HSS_2level_construct.o dirs
	$(LINK_EXECUTABLE)

.PHONY: clean
clean:
	for dir in $(DIRS) test; do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) bin/
