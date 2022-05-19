TOPSRCDIR = ../..
SOURCES :=  $(wildcard *.cpp)
sublib := libfranklin.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOPSRCDIR)/examples/franklin/include $< -c -o $@
