TOPSRCDIR = .
include $(TOPSRCDIR)/common.mk

DIRS := src/classes src/functions src/util examples/franklin
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

# non-distributed HSS code.
.PHONY: examples/franklin/HSS
examples/franklin/HSS:
	$(MAKE) -C $@

HSS_main : % : dirs examples/franklin/HSS
	$(CXX) libHSS_main.a libfranklin.a  $(OBJLIBS) $(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

# non-distributed H2 code
.PHONY: examples/franklin/H2
examples/franklin/H2:
	$(MAKE) -C $@

H2_main : % : dirs examples/franklin/H2
	$(CXX) libH2_main.a libfranklin.a  $(OBJLIBS) $(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

# slate rules
.PHONY: examples/franklin/HSS_slate
examples/franklin/HSS_slate:
	$(MAKE) -C $@

HSS_slate : % : dirs examples/franklin/HSS_slate
	$(MPICXX) libHSS_slate.a libfranklin.a  $(OBJLIBS) $(LDFLAGS) $(SLATE_LIB) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

# scalapack rules
.PHONY: examples/franklin/HSS_scalapack
examples/franklin/HSS_scalapack:
	$(MAKE) -C $@

HSS_scalapack : % : dirs examples/franklin/HSS_scalapack
	$(MPICXX) libHSS_scalapack.a libfranklin.a  $(OBJLIBS) $(LDFLAGS) $(SCALAPACK_LIB) -o $@; \
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
	for dir in $(DIRS) examples/franklin/HSS examples/franklin/H2 $(TEST) $(EXAMPLES); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) bin/ *.a

EIGEN_EXAMPLE_EXECUTABLES := Construction_SymmetricH2_Nlevel_with_sampling \
	LDL_SymmetricH2_Nlevel \
	LDL_SymmetricH2_Nlevel_nodependency \
	Eigen_Dense \
	Eigen_Dense_DSYEV \
	Eigen_Dense_DSYEVX \
	Eigen_SymmetricH2_Nlevel \
	Eigen_SymmetricH2_Nlevel_Ma2019 \
	Eigen_SymmetricH2_Nlevel_nodependency \
	Eigen_SymmetricH2_Nlevel_nodependency_old

$(EIGEN_EXAMPLE_EXECUTABLES) : % : $(EXAMPLES)/%.o dirs
	$(LINK_EXECUTABLE)

examples/ParEigen_SymmetricH2_Nlevel.o : examples/ParEigen_SymmetricH2_Nlevel.cpp
	$(MPICXX) $(CXXFLAGS) $< -c -o $@

examples/ParEigen_SymmetricH2_Nlevel_nodependency.o : examples/ParEigen_SymmetricH2_Nlevel_nodependency.cpp
	$(MPICXX) $(CXXFLAGS) $< -c -o $@

examples/MSParEigen_SymmetricH2_Nlevel.o : examples/MSParEigen_SymmetricH2_Nlevel.cpp
	$(MPICXX) $(CXXFLAGS) $< -c -o $@

examples/ParEigen_Dense_PDSYEV.o : examples/ParEigen_Dense_PDSYEV.cpp
	$(MPICXX) $(CXXFLAGS) $< -c -o $@

examples/ParEigen_Dense_PDSYEVX.o : examples/ParEigen_Dense_PDSYEVX.cpp
	$(MPICXX) $(CXXFLAGS) $< -c -o $@

examples/ParEigen_Dense_ELPA.o : examples/ParEigen_Dense_ELPA.cpp
	$(MPICXX) $(CXXFLAGS) $(ELPA_INCLUDE) $< -c -o $@

examples/ParEigen_TestMPI.o : examples/ParEigen_TestMPI.cpp
	$(MPICXX) $(CXXFLAGS) $< -c -o $@

ParEigen_SymmetricH2_Nlevel : % : examples/ParEigen_SymmetricH2_Nlevel.o dirs
	$(MPICXX) $< $(OBJLIBS) $(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

ParEigen_SymmetricH2_Nlevel_nodependency : % : examples/ParEigen_SymmetricH2_Nlevel_nodependency.o dirs
	$(MPICXX) $< $(OBJLIBS) $(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

MSParEigen_SymmetricH2_Nlevel : % : examples/MSParEigen_SymmetricH2_Nlevel.o dirs
	$(MPICXX) $< $(OBJLIBS) $(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

ParEigen_Dense_PDSYEV : % : examples/ParEigen_Dense_PDSYEV.o dirs
	$(MPICXX) $< $(OBJLIBS) $(LDFLAGS) $(SCALAPACK_LIB) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

ParEigen_Dense_PDSYEVX : % : examples/ParEigen_Dense_PDSYEVX.o dirs
	$(MPICXX) $< $(OBJLIBS) $(LDFLAGS) $(SCALAPACK_LIB) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

ParEigen_Dense_ELPA : % : examples/ParEigen_Dense_ELPA.o dirs
	$(MPICXX) $< $(OBJLIBS) $(LDFLAGS) $(SCALAPACK_LIB) $(ELPA_LIB) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

ParEigen_TestMPI : % : examples/ParEigen_TestMPI.o dirs
	$(MPICXX) $< $(OBJLIBS) $(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

PAR_EIGEN_EXAMPLE_EXECUTABLES := ParEigen_SymmetricH2_Nlevel \
	ParEigen_SymmetricH2_Nlevel_nodependency \
	MSParEigen_SymmetricH2_Nlevel \
	ParEigen_Dense_PDSYEV \
	ParEigen_Dense_PDSYEVX \
	ParEigen_Dense_ELPA \
	ParEigen_TestMPI

eigen: $(EIGEN_EXAMPLE_EXECUTABLES) $(PAR_EIGEN_EXAMPLE_EXECUTABLES)
