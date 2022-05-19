TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
# sublib := libHSS_slate.a

libHSS_slate.a:  $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^
	$(MV) $@ $(TOPSRCDIR)/$@

include $(TOPSRCDIR)/common.mk
