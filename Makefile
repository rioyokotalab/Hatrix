TOPSRCDIR = .
include $(TOPSRCDIR)/common.mk

DIRS := src/classes src/functions src/util examples/franklin \
	examples/franklin/HSS
OBJLIBS := libclasses.a libfunctions.a libutil.a
TEST := test
EXAMPLES := examples
TEST_EXECUTABLES := scale svd triangular_matmul arithmetics matmul lu \
	qr Matrix norms id
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
	UMV_strong_1level \
	BLR2_construct \
	H2_matrix_2level_construct \
	BLR2_strong_far_dense_construct \
	UMV_strong_1level_debug \
	H2_strong_N_level \
	UMV_strong_chained_product \
	UMV_strong_H2_3level_hss_chain_product \
	UMV_strong_H2_3level_chain_product \
	UMV_strong_H2_Nlevel \
	UMV_strong_1level_better_index \
	UMV_BLR2_far_dense \
	H2_far_dense_construct \
	UMV_H2_far_dense \
	svd_vs_id

EXAMPLE_DIR_EXECUTABLES := HSS_main

.PHONY: dirs $(DIRS)
dirs: $(DIRS)

all: $(TEST_EXECUTABLES) $(EXAMPLE_EXECUTABLES) $(EXAMPLE_DIR_EXECUTABLES) \
	HSS_slate HSS_scalapack

$(DIRS):
	$(MAKE) -C $@

LINK_EXECUTABLE = $(CXX) $< $(OBJLIBS) $(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/


# The extra colon is needed for correct expansion in the dependency list
# https://stackoverflow.com/questions/16262344/pass-a-target-name-to-dependency-list-in-makefile
$(TEST_EXECUTABLES): % : $(TEST)/%.o dirs
	$(LINK_EXECUTABLE)

$(EXAMPLE_EXECUTABLES) : % : $(EXAMPLES)/%.o dirs
	$(LINK_EXECUTABLE)

$(EXAMPLE_DIR_EXECUTABLES) : % : lib%.a dirs
	$(CXX) $< $(OBJLIBS) libfranklin.a $(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

# slate rules
.PHONY: examples/franklin/HSS_slate
examples/franklin/HSS_slate:
	$(MAKE) -C $@

HSS_slate : % : libHSS_slate.a dirs examples/franklin/HSS_slate
	$(MPICXX) $< $(OBJLIBS) libfranklin.a $(LDFLAGS) $(SLATE_LIB) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

# scalapack rules
.PHONY: examples/franklin/HSS_scalapack
examples/franklin/HSS_scalapack:
	$(MAKE) -C $@

HSS_scalapack : % : libHSS_scalapack.a dirs examples/franklin/HSS_scalapack
	$(MPICXX) $< $(OBJLIBS) libfranklin.a $(LDFLAGS) $(SCALAPACK_LIB) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

UMV_strong_H2_Nlevel_starsh: % : $(EXAMPLES)/%.o dirs
	$(LINK_EXECUTABLE)

test: $(TEST_EXECUTABLES)
	for e in $(TEST_EXECUTABLES); do \
		./bin/$$e; \
	done

.PHONY: clean
.SILENT: clean
clean:
	for dir in $(DIRS) $(TEST) $(EXAMPLES); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) bin/
