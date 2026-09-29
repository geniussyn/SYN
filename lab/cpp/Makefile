# The timing harnesses. Every reported millisecond comes from one of these: the
# Python bindings copy the query matrix and take the GIL around the results,
# which at ~0.03 ms/query is not a rounding error.
#
#   make hnsw_bench                      no external dependency
#   make FAISS_INC=/path/to/faiss-src    both
#
# hnswlib is vendored under third_party/ (header-only, MIT). faiss is not: the
# pip wheel `faiss-gpu-cu12` bundles the shared library but ships no headers, so
# FAISS_INC must point at a faiss source tree of the same version -- the wheel's
# version is `python3 -c "import faiss; print(faiss.__version__)"`:
#
#   git clone --depth 1 --branch v1.14.0 https://github.com/facebookresearch/faiss
#
# `conda install -c pytorch faiss-cpu` provides headers and library together and
# needs only FAISS_INC=$CONDA_PREFIX/include FAISS_LIBDIR=$CONDA_PREFIX/lib.
#
# --disable-new-dtags is not optional when linking the wheel's library: the
# default RUNPATH is not searched for a dependency's own dependencies, and
# libfaiss needs the bundled openblas beside it.
CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -std=c++17 -Wall
HNSW_INC ?= third_party
FAISS_INC ?=
FAISS_LIBDIR ?= $(shell python3 -c "import glob,site;c=[g for d in site.getsitepackages() for g in glob.glob(d+'/faiss*.libs')];print(c[0] if c else '')" 2>/dev/null)
FAISS_LIB ?= $(notdir $(wildcard $(FAISS_LIBDIR)/libfaiss*.so))
FAISS_LINK = -L$(FAISS_LIBDIR) -l:$(FAISS_LIB) -Wl,-rpath,$(FAISS_LIBDIR) -Wl,--disable-new-dtags

all: hnsw_bench faiss_bench nsg_bench

hnsw_bench: hnsw_bench.cpp
	$(CXX) $(CXXFLAGS) -I$(HNSW_INC) -o $@ $<

faiss_bench nsg_bench: %: %.cpp
	@test -n "$(FAISS_INC)" || { echo "set FAISS_INC -- see the header of this Makefile"; exit 1; }
	@test -n "$(FAISS_LIB)" || { echo "no libfaiss found; set FAISS_LIBDIR and FAISS_LIB"; exit 1; }
	$(CXX) $(CXXFLAGS) -I$(FAISS_INC) -o $@ $< $(FAISS_LINK)

clean:
	rm -f hnsw_bench faiss_bench nsg_bench
.PHONY: all clean
