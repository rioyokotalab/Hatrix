TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libHSS_slate.a

include $(TOPSRCDIR)/common.mk

SLATE_INCLUDE= -I/Users/sameer/gitrepos/slate/build/include  -I../ -DSLATE_NO_CUDA=1 \
	-DSLATE_NO_HIP=1

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(SLATE_INCLUDE) -I$(TOPSRCDIR)/examples/franklin/include $< -c -o $@
