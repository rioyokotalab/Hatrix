TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libH2_ptg.a

include $(TOPSRCDIR)/common.mk

hatrix_factorize.c :
	parsec-ptgpp hatrix_factorize.jdf

%.o : %.cpp hatrix_factorize.c
	$(MPICXX) $(CXXFLAGS) $(PARSEC_INCLUDE) -I$(TOPSRCDIR)/examples/distributed/include $< -c -o $@
