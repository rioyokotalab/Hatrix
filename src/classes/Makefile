TOPSRCDIR = ../..
include $(TOPSRCDIR)/make.inc


lib_classes := lib_classes.a
SOURCES     := Matrix.cpp
OBJECTS     := $(subst .cpp,.o,$(SOURCES)) 

.PHONY: $(lib_classes)
$(lib_classes): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

.PHONY: clean
clean:
	$(RM) -f $(OBJECTS) $(lib_classes)
