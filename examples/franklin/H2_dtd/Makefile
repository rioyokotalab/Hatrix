TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libH2_dtd.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(PARSEC_INCLUDE) -I$(TOPSRCDIR)/examples/franklin/include $< -c -o $@
