TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
JDF_SOURCES := h2_factorize.jdf
JDF_OUTPUTS := h2_factorize
sublib := libH2_ptg.a

include $(TOPSRCDIR)/common.mk

# include jdf_sources.mk
# JDF_COMPILED_SOURCES := $(subst .c,.o,$(AUTO_JDF_SRCS))

%.jdf :
	parsec-ptgpp -E -i $(JDF_SOURCES) -o $(JDF_OUTPUTS)

%.c : %.jdf
	parsec-ptgpp -E -i $(JDF_SOURCES) -o $(JDF_OUTPUTS)

%.o : %.c
	echo "hello C rule. $^"
	$(MPICC) $(CFLAGS) $(PARSEC_INCLUDE) $< -c -o $@

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(PARSEC_INCLUDE) -I$(TOPSRCDIR)/examples/distributed/include $< -c -o $@

.PHONY: clean
clean:
	$(RM) *.c *.h *.o
