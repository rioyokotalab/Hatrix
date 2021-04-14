TOPSRCDIR = ../..
include $(TOPSRCDIR)/make.inc


lib_classes := libclasses.a
SOURCES     := Matrix.cpp
OBJECTS     := $(subst .cpp,.o,$(SOURCES))

all: $(lib_classes)

$(lib_classes): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^
	$(MV) $@ $(TOPSRCDIR)/$@

.PHONY: clean
clean:
	$(RM) -f $(OBJECTS) $(lib_classes)
