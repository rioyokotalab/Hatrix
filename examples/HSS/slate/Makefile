TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libHSS_slate.a

include $(TOPSRCDIR)/common.mk

SLATE_INCLUDE= -I/Users/sameer/gitrepos/slate/build/include -DSLATE_NO_CUDA=1 -DSLATE_NO_HIP=1
CXX=mpicxx

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(SLATE_INCLUDE) $< -c -o $@
