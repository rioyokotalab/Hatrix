include $(TOPSRCDIR)/make.inc

OBJECTS := $(subst .cpp,.o,$(SOURCES))

all: $(sublib)

$(sublib): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^
	$(MV) $@ $(TOPSRCDIR)/$@

.PHONY: clean
clean:
	$(RM) -f $(OBJECTS) $(sublib)

