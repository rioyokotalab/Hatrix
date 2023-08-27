TOPSRCDIR = .
include $(TOPSRCDIR)/common.mk

DIRS := src/classes src/functions src/util examples/distributed
OBJLIBS := libclasses.a libfunctions.a libutil.a
EXAMPLES := examples

.PHONY: dirs $(DIRS)
dirs: $(DIRS)

$(DIRS):
	$(MAKE) -C $@

LINK_EXECUTABLE = $(CXX) $< $(OBJLIBS) $(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

# H2 matrix distributed construction.
.PHONY: examples/distributed/H2_construct
examples/distributed/H2_construct:
	$(MAKE) -C $@

H2_construct : % : dirs examples/distributed/H2_construct
	$(MPICXX) libH2_construct.a libdistributed.a $(OBJLIBS) $(LDFLAGS) \
	$(PARSEC_LIB) $(SCALAPACK_LIB) $(ELSES_OBJ) \
	-o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

# parsec H2 matrix
.PHONY: examples/distributed/H2_dtd
examples/distributed/H2_dtd:
	$(MAKE) -C $@

H2_dtd : % : dirs examples/distributed/H2_dtd
	$(MPICXX) libH2_dtd.a libdistributed.a $(OBJLIBS) $(LDFLAGS) $(PARSEC_LIB) $(SCALAPACK_LIB) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/


# parsec H2 matrix with PTG
.PHONY: examples/distributed/H2_ptg
examples/distributed/H2_ptg:
	$(MAKE) -C $@

H2_ptg : % : dirs examples/distributed/H2_ptg
	$(MPICXX) libH2_ptg.a libdistributed.a $(OBJLIBS) $(LDFLAGS) $(PARSEC_LIB) $(SCALAPACK_LIB) -o $@; \
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

# non-distributed H2 code
.PHONY: examples/distributed/H2_eigen
examples/distributed/H2_eigen:
	$(MAKE) -C $@

H2_eigen : % : dirs examples/distributed/H2_eigen
	$(CXX) libH2_eigen.a libdistributed.a $(OBJLIBS) \
	/home/sameer.deshmukh/ELSES_mat_calc-master/src/src.a \
	/home/sameer.deshmukh/ELSES_mat_calc-master/xmlf90-1.2g-elses/macros/lib/libflib.a \
	$(LDFLAGS) -o $@; \
	mkdir -p bin; \
	$(MV) $@ bin/

.PHONY: clean
.SILENT: clean
clean:
	for dir in $(DIRS)  \
		examples/distributed/H2 examples/distributed/H2_construct \
		examples/distributed/H2_dtd examples/distributed/H2_ptg \
		$(TEST) $(EXAMPLES); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
	$(RM) $(OBJLIBS) bin/ *.a
