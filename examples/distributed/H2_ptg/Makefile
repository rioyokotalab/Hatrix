TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libH2_ptg.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(PARSEC_INCLUDE) -I$(TOPSRCDIR)/examples/distributed/include $< -c -o $@
