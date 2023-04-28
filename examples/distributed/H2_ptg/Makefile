TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
JDF_SOURCES := h2_factorize.jdf
JDF_OUTPUTS := h2_factorize
sublib := libH2_ptg.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(PARSEC_INCLUDE) -I$(TOPSRCDIR)/examples/distributed/include $< -c -o $@

.PHONY: clean
clean:
	$(RM) *.c h2_factorize.h *.o
