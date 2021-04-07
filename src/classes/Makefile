TOPSRCDIR = ../..
include $(TOPSRCDIR)/make.inc

.PHONY: clean

.SUFFIXES: .cpp .cu
all: classes

CLASSES = Matrix.o

classes: $(CLASSES)

clean:
	$(RM) -f $(CLASSES)
