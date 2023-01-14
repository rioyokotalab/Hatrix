TOPSRCDIR = ../../..
# SOURCES :=  $(wildcard *.cpp)
SOURCES := globals.cpp h2_dtd_construction.cpp h2_dtd_factorize_tests.cpp h2_dtd_operations.cpp main.cpp h2_tasks.o
sublib := libH2_dtd.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(PARSEC_INCLUDE) -I$(TOPSRCDIR)/examples/franklin/include $< -c -o $@

h2_tasks.o : h2_tasks.cpp
	$(MPICXX) $(CXXFLAGS) -pg $(PARSEC_INCLUDE) -I$(TOPSRCDIR)/examples/franklin/include $< -c -o $@
