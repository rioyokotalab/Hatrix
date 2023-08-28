TOPSRCDIR = ../../..
JDF_SOURCES := h2_factorize_flows.jdf
JDF_OUTPUTS := h2_factorize_flows
SOURCES :=  $(wildcard *.cpp)
sublib := libH2_construct.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(PARSEC_INCLUDE) \
	-I$(TOPSRCDIR)/examples/distributed/include $< -c -o $@

.PHONY: clean
clean:
	$(RM) *.c h2_factorize_flows.h *.o
