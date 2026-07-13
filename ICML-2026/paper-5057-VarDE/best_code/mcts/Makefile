CXX = g++



#####
# Defining targets, sources and flags
#####

SRC_DIR = src
BIN_DIR = bin

COMMON_SOURCES = $(wildcard src/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/common/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/est/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/ments/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/ments/dents/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/ments/rents/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/ments/tents/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/varde/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/uct/*.cpp)
COMMON_SOURCES += $(wildcard src/distributions/*.cpp)
COMMON_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(COMMON_SOURCES))

# Shared environment implementations used by experiment binaries
ENV_SOURCES = $(wildcard src/env/*.cpp)
ENV_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(ENV_SOURCES))

# Parameter tuning main for FrozenLake
TUNE_SOURCES = src/exp/tune_frozenlake.cpp
TUNE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(TUNE_SOURCES))

# Parameter tuning main for Sailing
TUNE_SAILING_SOURCES = src/exp/tune_sailing.cpp
TUNE_SAILING_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(TUNE_SAILING_SOURCES))

# Parameter tuning main for Taxi
TUNE_TAXI_SOURCES = src/exp/tune_taxi.cpp
TUNE_TAXI_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(TUNE_TAXI_SOURCES))

# Parameter tuning main for Synthetic Tree
TUNE_STREE_SOURCES = src/exp/tune_stree.cpp
TUNE_STREE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(TUNE_STREE_SOURCES))

# FrozenLake experiment with tuned parameters
RUN_SOURCES = src/exp/run_frozenlake.cpp
RUN_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_SOURCES))

# Sailing experiment with tuned parameters
RUN_SAILING_SOURCES = src/exp/run_sailing.cpp
RUN_SAILING_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_SAILING_SOURCES))

# Taxi experiment with tuned parameters
RUN_TAXI_SOURCES = src/exp/run_taxi.cpp
RUN_TAXI_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_TAXI_SOURCES))

# Synthetic Tree experiment with tuned parameters
RUN_STREE_SOURCES = src/exp/run_stree.cpp
RUN_STREE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_STREE_SOURCES))

# VarDE debug main
VarDE_DBG_SOURCES = src/exp/debug_varde.cpp
VarDE_DBG_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(VarDE_DBG_SOURCES))
# DENTS debug main
DENTS_DBG_SOURCES = src/exp/debug_dents.cpp
DENTS_DBG_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(DENTS_DBG_SOURCES))
# VarDE Sailing debug main
VarDE_DBG_SAILING_SOURCES = src/exp/debug_varde_sailing.cpp
VarDE_DBG_SAILING_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(VarDE_DBG_SAILING_SOURCES))
# VarDE Taxi debug main
VarDE_DBG_TAXI_SOURCES = src/exp/debug_varde_taxi.cpp
VarDE_DBG_TAXI_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(VarDE_DBG_TAXI_SOURCES))
# DENTS Sailing debug main
DENTS_DBG_SAILING_SOURCES = src/exp/debug_dents_sailing.cpp
DENTS_DBG_SAILING_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(DENTS_DBG_SAILING_SOURCES))
# DENTS Taxi debug main
DENTS_DBG_TAXI_SOURCES = src/exp/debug_dents_taxi.cpp
DENTS_DBG_TAXI_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(DENTS_DBG_TAXI_SOURCES))

# VarDE Synthetic Tree debug main
VarDE_DBG_STREE_SOURCES = src/exp/debug_varde_stree.cpp
VarDE_DBG_STREE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(VarDE_DBG_STREE_SOURCES))
# DENTS Synthetic Tree debug main
DENTS_DBG_STREE_SOURCES = src/exp/debug_dents_stree.cpp
DENTS_DBG_STREE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(DENTS_DBG_STREE_SOURCES))

INCLUDES = -Iinclude/ -Isrc/ -I.

CPPFLAGS = $(INCLUDES) -Wall -std=c++20
CPPFLAGS_DEBUG = -g

LDFLAGS = -lpthread

TARGET_MCTS = mcts
TARGET_MCTS_TUNE = mcts-tune-frozenlake
TARGET_MCTS_TUNE_SAILING = mcts-tune-sailing
TARGET_MCTS_TUNE_TAXI = mcts-tune-taxi
TARGET_MCTS_TUNE_STREE = mcts-tune-stree
TARGET_MCTS_RUN = mcts-run-frozenlake
TARGET_MCTS_RUN_SAILING = mcts-run-sailing
TARGET_MCTS_RUN_TAXI = mcts-run-taxi
TARGET_MCTS_RUN_STREE = mcts-run-stree
TARGET_MCTS_VarDE_DBG = mcts-debug-varde
TARGET_MCTS_DENTS_DBG = mcts-debug-dents
TARGET_MCTS_VarDE_DBG_SAILING = mcts-debug-varde-sailing
TARGET_MCTS_DENTS_DBG_SAILING = mcts-debug-dents-sailing
TARGET_MCTS_VarDE_DBG_TAXI = mcts-debug-varde-taxi
TARGET_MCTS_DENTS_DBG_TAXI = mcts-debug-dents-taxi
TARGET_MCTS_VarDE_DBG_STREE = mcts-debug-varde-stree
TARGET_MCTS_DENTS_DBG_STREE = mcts-debug-dents-stree



#####
# Default target
#####

# Default, build common objects
all: $(TARGET_MCTS)



#####
# (Custom) Rules to build object files
# Adapted from: https://stackoverflow.com/questions/41568508/makefile-compile-multiple-c-file-at-once/41924169#41924169
# N.B. A rule of the form "some_dir/%.o: bin/some_dir/%.cpp" for some reason makes make recompile all files everytime
#####

# Required to enable use of $$
.SECONDEXPANSION:

## General rule formula for building object files 
## N.B. $(@D) gets the directory of the file
#$(OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
#	@mkdir -p $(@D)
#	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build object files rule
$(COMMON_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build environment object files rule
$(ENV_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build tuner object files rule
$(TUNE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build sailing tuner object files rule
$(TUNE_SAILING_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build taxi tuner object files rule
$(TUNE_TAXI_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build synthetic tree tuner object files rule
$(TUNE_STREE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build run object files rule
$(RUN_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build sailing run object files rule
$(RUN_SAILING_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build taxi run object files rule
$(RUN_TAXI_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build synthetic tree run object files rule
$(RUN_STREE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build VarDE debug object files rule
$(VarDE_DBG_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build DENTS debug object files rule
$(DENTS_DBG_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build VarDE Sailing debug object files rule
$(VarDE_DBG_SAILING_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build VarDE Taxi debug object files rule
$(VarDE_DBG_TAXI_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build DENTS Sailing debug object files rule
$(DENTS_DBG_SAILING_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build DENTS Taxi debug object files rule
$(DENTS_DBG_TAXI_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build VarDE Synthetic Tree debug object files rule
$(VarDE_DBG_STREE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build DENTS Synthetic Tree debug object files rule
$(DENTS_DBG_STREE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<



#####
# Targets
#####

# Compiling 'mcts' just builds objects for now (to be used as prereq basically)
$(TARGET_MCTS): $(COMMON_OBJECTS)

# Build FrozenLake tuner
$(TARGET_MCTS_TUNE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(TUNE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Sailing tuner
$(TARGET_MCTS_TUNE_SAILING): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(TUNE_SAILING_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Taxi tuner
$(TARGET_MCTS_TUNE_TAXI): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(TUNE_TAXI_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Synthetic Tree tuner
$(TARGET_MCTS_TUNE_STREE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(TUNE_STREE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build FrozenLake experiment with tuned parameters
$(TARGET_MCTS_RUN): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Sailing experiment with tuned parameters
$(TARGET_MCTS_RUN_SAILING): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_SAILING_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Taxi experiment with tuned parameters
$(TARGET_MCTS_RUN_TAXI): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_TAXI_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Synthetic Tree experiment with tuned parameters
$(TARGET_MCTS_RUN_STREE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_STREE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build VarDE debug binary
$(TARGET_MCTS_VarDE_DBG): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(VarDE_DBG_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build DENTS debug binary
$(TARGET_MCTS_DENTS_DBG): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(DENTS_DBG_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build VarDE Sailing debug binary
$(TARGET_MCTS_VarDE_DBG_SAILING): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(VarDE_DBG_SAILING_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build VarDE Taxi debug binary
$(TARGET_MCTS_VarDE_DBG_TAXI): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(VarDE_DBG_TAXI_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build DENTS Sailing debug binary
$(TARGET_MCTS_DENTS_DBG_SAILING): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(DENTS_DBG_SAILING_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build DENTS Taxi debug binary
$(TARGET_MCTS_DENTS_DBG_TAXI): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(DENTS_DBG_TAXI_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build VarDE Synthetic Tree debug binary
$(TARGET_MCTS_VarDE_DBG_STREE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(VarDE_DBG_STREE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build DENTS Synthetic Tree debug binary
$(TARGET_MCTS_DENTS_DBG_STREE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(DENTS_DBG_STREE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)




#####
# Clean
#####

# Clean up compiled files
# "[X] && Y || Z" is bash for if X then Y else Z 
# [ -e <filename> ] checks if filename exists
# : is a no-op 
clean:
	@rm -rf $(BIN_DIR) > /dev/null 2> /dev/null
	@[ -f $(TARGET_MCTS_TUNE) ] && @rm $(TARGET_MCTS_TUNE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_TUNE_SAILING) ] && @rm $(TARGET_MCTS_TUNE_SAILING) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_TUNE_TAXI) ] && @rm $(TARGET_MCTS_TUNE_TAXI) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_TUNE_STREE) ] && @rm $(TARGET_MCTS_TUNE_STREE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN) ] && @rm $(TARGET_MCTS_RUN) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_SAILING) ] && @rm $(TARGET_MCTS_RUN_SAILING) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_TAXI) ] && @rm $(TARGET_MCTS_RUN_TAXI) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_STREE) ] && @rm $(TARGET_MCTS_RUN_STREE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_VarDE_DBG) ] && @rm $(TARGET_MCTS_VarDE_DBG) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_DENTS_DBG) ] && @rm $(TARGET_MCTS_DENTS_DBG) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_VarDE_DBG_SAILING) ] && @rm $(TARGET_MCTS_VarDE_DBG_SAILING) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_DENTS_DBG_SAILING) ] && @rm $(TARGET_MCTS_DENTS_DBG_SAILING) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_VarDE_DBG_STREE) ] && @rm $(TARGET_MCTS_VarDE_DBG_STREE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_DENTS_DBG_STREE) ] && @rm $(TARGET_MCTS_DENTS_DBG_STREE) > /dev/null 2> /dev/null || :


#####
# Phony targets, so make knows when a target isn't producing a corresponding output file of same name
#####
.PHONY: clean $(TARGET_MCTS) $(TARGET_MCTS_TUNE) $(TARGET_MCTS_TUNE_SAILING) $(TARGET_MCTS_TUNE_TAXI) $(TARGET_MCTS_TUNE_STREE) $(TARGET_MCTS_RUN) $(TARGET_MCTS_RUN_SAILING) $(TARGET_MCTS_RUN_TAXI) $(TARGET_MCTS_RUN_STREE) $(TARGET_MCTS_VarDE_DBG) $(TARGET_MCTS_DENTS_DBG) $(TARGET_MCTS_VarDE_DBG_SAILING) $(TARGET_MCTS_DENTS_DBG_SAILING) $(TARGET_MCTS_VarDE_DBG_TAXI) $(TARGET_MCTS_DENTS_DBG_TAXI) $(TARGET_MCTS_VarDE_DBG_STREE) $(TARGET_MCTS_DENTS_DBG_STREE)
