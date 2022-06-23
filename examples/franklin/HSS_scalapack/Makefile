TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libHSS_scalapack.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) -I$(TOPSRCDIR)/examples/franklin/include $< -c -o $@
