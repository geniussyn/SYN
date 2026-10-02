# Rebuild every figure from the shipped TSV. Needs gnuplot >= 5.2 with pdfcairo.
#
#   make          every figure
#   make check    every script runs and leaves a non-empty PDF
#   make clean    remove the PDFs, keep the data
#
# The data is the artifact. These scripts are a reference rendering and are not
# the paper's typography -- the paper's panels come from the matplotlib scripts
# in analysis/, which read the same results and are the source of record.

# paper_*.gp assemble the per-panel sheets into the figures the paper prints;
# the rest are the single panels those are built from.
GP  := $(wildcard *.gp)
PAPER := $(wildcard paper_*.gp)
PDF := $(GP:.gp=.pdf)

all: $(PDF)

paper: $(PAPER:.gp=.pdf)

%.pdf: %.gp
	gnuplot $<

check: all
	@bad=0; for f in $(PDF); do \
	  if [ ! -s "$$f" ]; then echo "EMPTY  $$f"; bad=$$((bad+1)); fi; \
	done; \
	echo "$(words $(PDF)) figure(s), $$bad empty"; \
	[ $$bad -eq 0 ]

clean:
	rm -f $(PDF)

.PHONY: all paper check clean
