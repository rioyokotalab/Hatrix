TOPSRCDIR = ..
include $(TOPSRCDIR)/make.inc

lib_test = lib_test.a
SOURCES = gemm.cpp
OBJECTS = $(subst .cpp,.o,$(SOURCES))

$(lib_test): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

.PHONY: clean
clean:
	$(RM) -f $(OBJECTS) $(lib_test)
