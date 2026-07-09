# ===============================
# QKMEANS - Fast k-means++ Seeding
# ===============================

# Compiler and flags
CXX       := g++
CXXFLAGS  := -O3 -march=native -ffast-math -std=c++17
OMPFLAGS  := -fopenmp
INCLUDES  := -I.

# Directories
BIN_DIR   := bin
SRC_DIR   := src/bin

# Headers
ALGO_HEADERS := $(wildcard src/algorithms/*.hpp)
CORE_HEADERS := $(wildcard src/core/*.hpp)
HEADERS := $(ALGO_HEADERS) $(CORE_HEADERS)

# ===============================
# Main targets (new unified runners)
# ===============================
MAIN_TARGETS := $(BIN_DIR)/run_single \
                $(BIN_DIR)/run_comparison \
                $(BIN_DIR)/run_sweep

# ===============================
# Default: build all targets
# ===============================
all: $(BIN_DIR) $(MAIN_TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# ===============================
# Main unified runners
# ===============================
$(BIN_DIR)/run_single: $(SRC_DIR)/run_single.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) $< -o $@

$(BIN_DIR)/run_comparison: $(SRC_DIR)/run_comparison.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) $< -o $@

$(BIN_DIR)/run_sweep: $(SRC_DIR)/run_sweep.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) $< -o $@

# ===============================
# Debug build
# ===============================
debug: $(BIN_DIR)
	$(CXX) -O0 -g -std=c++17 $(OMPFLAGS) $(INCLUDES) $(SRC_DIR)/run_single.cpp -o $(BIN_DIR)/run_single_debug
	$(CXX) -O0 -g -std=c++17 $(OMPFLAGS) $(INCLUDES) $(SRC_DIR)/run_comparison.cpp -o $(BIN_DIR)/run_comparison_debug

# ===============================
# Generate configs
# ===============================
configs:
	python3 configs/generate_configs.py

# ===============================
# Run benchmarks
# ===============================
run-mnist: $(BIN_DIR)/run_comparison
	./$(BIN_DIR)/run_comparison configs/generated/comparison_mnist.json

run-cifar10: $(BIN_DIR)/run_comparison
	./$(BIN_DIR)/run_comparison configs/generated/comparison_cifar10.json

# ===============================
# Clean
# ===============================
clean:
	rm -rf $(BIN_DIR)

clean-configs:
	rm -rf configs/generated

clean-all: clean clean-configs
	rm -rf results/*.csv

# ===============================
# Help
# ===============================
help:
	@echo "QKMEANS - Fast k-means++ Seeding"
	@echo ""
	@echo "Main targets:"
	@echo "  make                  -> Build unified runners"
	@echo ""
	@echo "Binaries:"
	@echo "  bin/run_single        -> Run single algorithm: ./bin/run_single <algo> config.json"
	@echo "  bin/run_comparison    -> Run all algorithms:   ./bin/run_comparison config.json"
	@echo "  bin/run_sweep         -> Hyperparameter sweep: ./bin/run_sweep config.json"
	@echo ""
	@echo "Utilities:"
	@echo "  make configs          -> Generate config files from template"
	@echo "  make debug            -> Build with debug symbols"
	@echo "  make clean            -> Remove bin/ directory"
	@echo ""
	@echo "Algorithms: kmeanspp, afkmc2, prone, pronecoreset, fastcoreset, rejectionlsh, qkmeans"

.PHONY: all debug clean clean-configs clean-all help configs run-mnist run-cifar10
