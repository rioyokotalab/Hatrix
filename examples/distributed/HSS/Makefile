TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libHSS_main.a

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOPSRCDIR)/examples/distributed/include $< -c -o $@
