TOPSRCDIR = .
include $(TOPSRCDIR)/make.inc

DIRS := src/classes src/functions src/util
OBJLIBS := libfunctions.a libclasses.a libutil.a
TEST := test
EXAMPLES := examples
TEST_EXECUTABLES := scale svd triangular_matmul arithmetics matmul lu \
	qr Matrix norms
EXAMPLE_EXECUTABLES := 2x2_BlockDense_LU \
	2x2_BlockDense_QR \
	2x2_BLR_LU \
	2x2_BLR_weak_GEMM \
	HSS_2level_construct \
	HSS_Nlevel_construct \
	HSS_Nlevel_construct_chandrasekaran \
	HSS_Nlevel_construct_miro \
	NxN_BlockDense_LU \
	NxN_BLR_LU \
	NxN_BLR_strong_GEMM \
	NxN_BLR_strong_LU \
	NxN_BLR_weak_GEMM \
	UMV_weak_1level \
	UMV_weak_1level_noblock \
	UMV_weak_2level \
	UMV_weak_Nlevel \
	UMV_strong_1level

.PHONY: dirs $(DIRS)
dirs: $(DIRS)

all: $(TEST_EXECUTABLES) $(EXAMPLE_EXECUTABLES)

$(DIRS):
	$(MAKE) -C $@

LINK_EXECUTABLE = $(CXX) $< $(OBJLIBS) $(LDFLAGS)  -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

# The extra colon is needed for correct expansion in the dependency list
# https://stackoverflow.com/questions/16262344/pass-a-target-name-to-dependency-list-in-makefile
$(TEST_EXECUTABLES): % : $(TEST)/%.o dirs
	$(LINK_EXECUTABLE)

$(EXAMPLE_EXECUTABLES) : % : $(EXAMPLES)/%.o dirs
	$(LINK_EXECUTABLE)

test: $(TEST_EXECUTABLES)
	for e in $(TEST_EXECUTABLES); do \
		./bin/$$e; \
	done

.PHONY: clean
clean:
	for dir in $(DIRS) $(TEST) $(EXAMPLES); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) bin/
