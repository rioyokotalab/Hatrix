TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libHSS_dtd.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(PARSEC_INCLUDE) -I$(TOPSRCDIR)/examples/distributed/include $< -c -o $@
