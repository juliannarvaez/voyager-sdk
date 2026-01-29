# Copyright Axelera AI, 2025
# Makefile for C++ Rowing Ergometer Recorder

CXX = g++
CXXFLAGS = -std=c++20 -O3 -march=native -mtune=native -flto
CXXFLAGS += -Wall -Wextra -Wno-unused-parameter
CXXFLAGS += -pthread

# Performance optimizations
CXXFLAGS += -DNDEBUG  # Disable asserts
CXXFLAGS += -ffast-math  # Aggressive math optimizations
CXXFLAGS += -funroll-loops
CXXFLAGS += -finline-functions

# Cache optimization
CXXFLAGS += -falign-functions=64
CXXFLAGS += -falign-loops=64

# Link-time optimization
LDFLAGS = -flto -pthread
LIBS = -lm

# Axelera SDK paths (for integration - optional for standalone build)
# AXELERA_SDK = /voyager-sdk
# AXELERA_INCLUDE = -I$(AXELERA_SDK)/operators/axstreamer/include
# AXELERA_INCLUDE += -I$(AXELERA_SDK)/operators/gstaxstreamer/include
# AXELERA_LIBS = -L$(AXELERA_SDK)/lib -Wl,-rpath,$(AXELERA_SDK)/lib

# Standalone build (no Axelera SDK dependency)
AXELERA_INCLUDE =
AXELERA_LIBS =

# OpenCV (optional - only if using Axelera integration)
# OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
# OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)
OPENCV_CFLAGS =
OPENCV_LIBS =

# Target executable
TARGET = rowing_recorder_cpp

# Source files
SOURCES = rowing_recorder_cpp.cpp keypoint_recorder.cpp phase_controller.cpp
HEADERS = keypoint_recorder.hpp phase_controller.hpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(AXELERA_LIBS) $(OPENCV_LIBS) $(LIBS)
	@echo "Build complete: $@"
	@echo ""
	@echo "Performance optimizations enabled:"
	@echo "  - O3 optimization with LTO"
	@echo "  - Native architecture tuning"
	@echo "  - Fast math and loop unrolling"
	@echo "  - Cache-aligned functions/loops"

%.o: %.cpp $(HEADERS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(AXELERA_INCLUDE) $(OPENCV_CFLAGS) -c $< -o $@

# Debug build (with symbols and asserts)
debug: CXXFLAGS = -std=c++17 -O0 -g -march=native
debug: CXXFLAGS += -Wall -Wextra -Wpedantic -pthread
debug: LDFLAGS = -pthread
debug: clean $(TARGET)
	@echo "Debug build complete"

# Profile-guided optimization build (two-phase)
pgo-generate:
	$(MAKE) clean
	$(MAKE) CXXFLAGS="$(CXXFLAGS) -fprofile-generate" LDFLAGS="$(LDFLAGS) -fprofile-generate"
	@echo ""
	@echo "PGO instrumented build complete."
	@echo "Run the program with typical workload, then run 'make pgo-use'"

pgo-use:
	$(MAKE) clean
	$(MAKE) CXXFLAGS="$(CXXFLAGS) -fprofile-use -fprofile-correction" LDFLAGS="$(LDFLAGS) -fprofile-use"
	@echo "PGO optimized build complete"

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -f *.gcda *.gcno  # PGO data
	@echo "Clean complete"

# Install to system
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/
	@echo "Installed to /usr/local/bin/$(TARGET)"

# Benchmark target
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	./$(TARGET) --source /dev/video0 --stats-interval 60 --no-progress --headless

# Show compiler optimizations
show-opts:
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	$(CXX) -Q --help=optimizers $(CXXFLAGS) | grep enabled

# Generate assembly for inspection
asm: keypoint_recorder.cpp
	$(CXX) $(CXXFLAGS) $(AXELERA_INCLUDE) -S -fverbose-asm -o keypoint_recorder.s keypoint_recorder.cpp
	@echo "Assembly output: keypoint_recorder.s"

.PHONY: all debug clean install benchmark show-opts pgo-generate pgo-use asm
