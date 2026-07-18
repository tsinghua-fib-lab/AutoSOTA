CXX ?= g++
SRC_DIR := src
BIN_DIR := bin
LOG_DIR := logs

PKG_CONFIG ?= pkg-config

ifeq ($(shell $(PKG_CONFIG) --exists simgrid && echo yes),yes)
SIMGRID_CFLAGS := $(shell $(PKG_CONFIG) --cflags simgrid)
SIMGRID_LIBS := $(shell $(PKG_CONFIG) --libs simgrid)
else
$(error simgrid not found via $(PKG_CONFIG). Install SimGrid >= 4.0 and ensure simgrid.pc is on PKG_CONFIG_PATH)
endif

# SimGrid headers transitively include Boost; simgrid.pc doesn't carry that
# path on Homebrew. Override with BOOST_CFLAGS=-I... if auto-detect misses.
BOOST_CFLAGS ?=
ifeq ($(BOOST_CFLAGS),)
  ifeq ($(shell uname -s),Darwin)
    BREW_BOOST := $(shell command -v brew >/dev/null 2>&1 && brew --prefix boost 2>/dev/null)
    ifneq ($(BREW_BOOST),)
      BOOST_CFLAGS := -I$(BREW_BOOST)/include
    endif
  endif
endif

CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 $(SIMGRID_CFLAGS) $(BOOST_CFLAGS)
LDFLAGS += $(SIMGRID_LIBS)

DP_BIN := $(BIN_DIR)/dp
DPR_BIN := $(BIN_DIR)/dpr
SSPARE_BIN := $(BIN_DIR)/spare

DP_SRC := $(SRC_DIR)/DP_DES.cpp $(SRC_DIR)/utility.cpp
DPR_SRC := $(SRC_DIR)/DPR_DES.cpp $(SRC_DIR)/utility.cpp
SSPARE_SRC := $(SRC_DIR)/SPARe_DES.cpp $(SRC_DIR)/utility.cpp

.PHONY: all clean

all: $(DP_BIN) $(DPR_BIN) $(SSPARE_BIN)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)
	mkdir -p $(LOG_DIR)

$(DP_BIN): $(DP_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(DPR_BIN): $(DPR_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(SSPARE_BIN): $(SSPARE_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(LOG_DIR)
