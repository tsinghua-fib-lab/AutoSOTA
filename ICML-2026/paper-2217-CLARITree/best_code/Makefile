CXX := g++
CXXFLAGS := -std=gnu++17 -O3 -fopenmp
INCLUDES := -I/usr/include/eigen3 -Iinclude
ARCH ?= -march=haswell -mtune=skylake
SRC := src/clari_tree.cpp src/clari_tree_const.cpp src/main.cpp
BIN := run_tree

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(BIN)

test: $(BIN)
	./$(BIN) data/auction/auction_continuous.csv 4.0 0.001 0.001 20 quantile
clean:
	rm -f $(BIN)
