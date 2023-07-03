TOPSRCDIR = ../../..
SOURCES :=  $(wildcard *.cpp)
sublib := libH2_eigen.a

ELSES_DIR=/home/sameer.deshmukh/ELSES_mat_calc-master

include $(TOPSRCDIR)/common.mk

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOPSRCDIR)/examples/distributed/include -I$(ELSES_DIR) $< -c -o $@
