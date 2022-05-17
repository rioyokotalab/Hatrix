TOPSRCDIR = ../..
SOURCES :=  $(wildcard *.cpp)

include $(TOPSRCDIR)/common.mk

libHSS_main.a:  $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^
	$(MV) $@ $(TOPSRCDIR)/$@
