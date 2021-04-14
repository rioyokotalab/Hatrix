TOPSRCDIR = ../..
include $(TOPSRCDIR)/make.inc

lib_functions = libfunctions.a
SOURCES = blas.cpp lapack.cpp
OBJECTS = $(subst .cpp,.o,$(SOURCES))

$(lib_functions): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

.PHONY: clean
clean:
	$(RM) -f $(OBJECTS) $(lib_functions)
