TOPSRCDIR = .
include $(TOPSRCDIR)/common.mk

DIRS := src/classes src/functions src/util examples/distributed
OBJLIBS := libclasses.a libfunctions.a libutil.a
TEST := test
EXAMPLES := examples
TEST_EXECUTABLES := scale svd triangular_matmul arithmetics matmul lu \
	qr Matrix norms id cholesky lu_solve
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
	svd_vs_id \
	UMV_H2_Nlevel


.PHONY: dirs $(DIRS)
dirs: $(DIRS)

all: $(TEST_EXECUTABLES) $(EXAMPLE_EXECUTABLES) $(EXAMPLE_DIR_EXECUTABLES)

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

# parsec H2 matrix
.PHONY: examples/distributed/H2_dtd
examples/distributed/H2_dtd:
	$(MAKE) -C $@

H2_dtd : % : dirs examples/distributed/H2_dtd
	$(MPICXX) libH2_dtd.a libdistributed.a $(OBJLIBS) $(LDFLAGS) $(PARSEC_LIB) $(SCALAPACK_LIB) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

# non-distributed HSS code.
.PHONY: examples/distributed/HSS
examples/distributed/HSS:
	$(MAKE) -C $@

HSS_main : % : dirs examples/distributed/HSS
	$(CXX) libHSS_main.a libdistributed.a  $(OBJLIBS) $(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

# non-distributed H2 code
.PHONY: examples/distributed/H2
examples/distributed/H2:
	$(MAKE) -C $@

H2_main : % : dirs examples/distributed/H2
	$(CXX) libH2_main.a libdistributed.a  $(OBJLIBS) $(LDFLAGS) -o $@; \
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
	for dir in $(DIRS) examples/distributed/HSS examples/distributed/H2 examples/distributed/H2_dtd $(TEST) $(EXAMPLES); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) bin/ *.a
