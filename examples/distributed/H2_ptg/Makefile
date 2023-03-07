TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
JDF_SOURCES := $(wildcard *.jdf)
sublib := libH2_ptg.a

include $(TOPSRCDIR)/common.mk

include jdf_sources.mk
JDF_COMPILED_SOURCES := $(subst .c,.o,$(AUTO_JDF_SRCS))

%.jdf :
	parsec-ptgpp $(JDF_SOURCES)
	echo "AUTO_JDF_SRCS=`echo *.c`" > jdf_sources.mk

jdf_sources.mk : %.jdf

%.o : %.cpp %.c
	$(MPICXX) $(CXXFLAGS) $(PARSEC_INCLUDE) -I$(TOPSRCDIR)/examples/distributed/include $< -c -o $@
