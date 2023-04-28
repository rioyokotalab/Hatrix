TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libDense.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOPSRCDIR)/examples/distributed/include $< -c -o $@
