TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

DIRS := src/classes src/functions src/util
OBJLIBS := libutil.a libclasses.a libfunctions.a
TEST := test
EXAMPLES := examples
EXECUTABLES := matmul lu qr Matrix HSS_2level_construction Qsparse_weak_1level

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

HSS_2level_construction: $(EXAMPLES)/HSS_2level_construction.o dirs
	$(LINK_EXECUTABLE)

Qsparse_weak_1level: $(EXAMPLES)/Qsparse_weak_1level.o dirs
	$(LINK_EXECUTABLE)


.PHONY: clean
clean:
	for dir in $(DIRS) test; do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) bin/
