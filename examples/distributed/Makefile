TOPSRCDIR = ../..
SOURCES :=  $(wildcard *.cpp)
sublib := libdistributed.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) -I$(TOPSRCDIR)/examples/distributed/include $< -c -o $@
