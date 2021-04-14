TOPSRCDIR = ../..
include $(TOPSRCDIR)/make.inc

lib_functions = lib_functions.a
SOURCES = blas.cpp lapack.cpp
OBJECTS = $(subst .cpp,.o,$(SOURCES))

$(lib_functions): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

.PHONY: clean
clean:
	$(RM) -f $(OBJECTS) $(lib_functions)
