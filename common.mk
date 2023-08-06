include $(TOPSRCDIR)/make.fugaku.inc

OBJECTS := $(subst .cpp,.o,$(SOURCES)) $(subst .c,.o,$(wildcard *.c))

all: $(sublib)

# The OBJECTS are files that are within the project that are compiled
# into object files from CPP source files.
$(sublib): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^
	$(MV) $@ $(TOPSRCDIR)/$@

.PHONY: clean
clean:
	$(RM) -f $(OBJECTS) $(sublib)
