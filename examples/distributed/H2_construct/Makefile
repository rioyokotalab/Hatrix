TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libH2_construct.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) -I$(TOPSRCDIR)/examples/distributed/include -I$(ELSES_ROOT) $< -c -o $@
