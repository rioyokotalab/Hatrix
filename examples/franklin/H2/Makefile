TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libH2_main.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOPSRCDIR)/examples/franklin/include $< -c -o $@
