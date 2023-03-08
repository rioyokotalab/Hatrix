TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
JDF_SOURCES := $(wildcard *.jdf)
JDF_OUTPUTS := $(basename $(JDF_SOURCES))
sublib := libH2_ptg.a

include $(TOPSRCDIR)/common.mk

include jdf_sources.mk
JDF_COMPILED_SOURCES := $(subst .c,.o,$(AUTO_JDF_SRCS))

%.jdf :
	parsec-ptgpp -E -i $(JDF_SOURCES) -o $(JDF_OUTPUTS)
	echo "AUTO_JDF_SRCS=`echo *.c`" > jdf_sources.mk

jdf_sources.mk : %.jdf

%.o : %.cpp h2_factorize.c
	$(MPICXX) $(CXXFLAGS) $(PARSEC_INCLUDE) -I$(TOPSRCDIR)/examples/distributed/include $^ -c -o $@

.PHONY: clean

clean:
	rm jdf_sources.mk
	rm *.c
	rm *.h
