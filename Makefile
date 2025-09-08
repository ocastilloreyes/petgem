# PETGEM Makefile

# === Paths and Settings ===

# Target executable
TARGET := build/kernel
# List of source files
SRCS := src/kernel.c src/common.c src/inputs.c src/transmitter.c src/grid.c src/assembly.c src/hvfem.c src/solver.c src/postprocessing.c
# List of object files
OBJS := $(SRCS:.c=.o)

# Our include folder
I_CFLAGS := -Iinclude

# PETSc variables
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# === Optional Extrae Support ===
# Conditional flag for Extrae support ( set 1 to include Extrae, 0 to exclude)
USE_EXTRAE ?= 0

# Add Extrae includes and flags if USE_EXTRAE is set to 1
ifeq ($(USE_EXTRAE), 1)
    E_CFLAGS := -I$(EXTRAE_HOME)/include -DUSE_EXTRAE
    E_LDFLAGS := -L$(EXTRAE_HOME)/lib -lmpitrace
endif

# === Build Target ===
all: build $(TARGET) # Asegura que 'build' existe antes de compilar

# Compile all object files and generate the final executable
$(TARGET): $(OBJS)
	$(CLINKER) $^ -o $@ $(CFLAGS) $(E_LDFLAGS) $(PETSC_LIB)

# Rule to compile each source file (uses PETSc's makefile variable)
# These objects are for the main application, placed in their source directories.
%.o: %.c
	${PETSC_COMPILE_SINGLE} $(CFLAGS) $(I_CFLAGS) $(E_FLAGS) $< -o $@

# Create build directory
build:
	@mkdir -p build

# === Unit Tests ===
UNITY_DIR = /opt/unity/src
TEST_SRCS := $(wildcard tests/test_*.c)
TEST_BIN := build/test_runner

# Source file for Unity framework (it's compiled along with tests)
TEST_UNITY_SRC := $(UNITY_DIR)/unity.c

# Application source files that contain a 'main' function for the main executable.
APP_MAIN_SRCS := src/kernel.c

# Application source files used by the test runner (excluding the main app source).
APP_SRCS_FOR_TEST_BUILD := $(filter-out $(APP_MAIN_SRCS), $(SRCS))

# --- Test Compilation and Linker Flags ---
# Include flags for PETSc, our project's include, and Unity
_TEST_INCLUDE_FLAGS := $(PETSC_CC_INCLUDES) -Iinclude -I$(UNITY_DIR)

# Coverage flags (compile and link)
_COVERAGE_COMPILE_FLAGS := -fprofile-arcs -ftest-coverage -O0
_COVERAGE_LINK_FLAGS := -lgcov --coverage

# Unity specific configuration flags
_UNITY_CONFIG_FLAGS := -DUNITY_INCLUDE_DOUBLE # For double support in Unity

# Combined CFLAGS for compiling test source files
TEST_SPECIFIC_CFLAGS := $(_TEST_INCLUDE_FLAGS) $(_COVERAGE_COMPILE_FLAGS) $(_UNITY_CONFIG_FLAGS)

# --- Objects for the Test Runner (compiled into build/) ---
# Convert .c paths to .o paths within the build/ directory
TEST_OBJ_FILES := $(patsubst tests/%.c,build/%.o,$(TEST_SRCS))
APP_SOURCE_TEST_OBJ_FILES := $(patsubst src/%.c,build/%.o,$(APP_SRCS_FOR_TEST_BUILD))
UNITY_OBJ_FILE := build/unity.o

# All objects that go into the test runner executable
ALL_TEST_RUNNER_OBJS := $(TEST_OBJ_FILES) $(APP_SOURCE_TEST_OBJ_FILES) $(UNITY_OBJ_FILE)

# Rules to compile source files for the test runner into 'build/'
build/%.o: tests/%.c | build
	@echo "  [TESTS] Compiling $< to $@"
	$(CLINKER) $(TEST_SPECIFIC_CFLAGS) $(I_CFLAGS) -c $< -o $@

build/%.o: src/%.c | build
	@echo "  [TESTS] Compiling $< to $@"
	$(CLINKER) $(TEST_SPECIFIC_CFLAGS) $(I_CFLAGS) -c $< -o $@

build/unity.o: $(UNITY_DIR)/unity.c | build
	@echo "  [TESTS] Compiling $< to $@"
	$(CLINKER) $(TEST_SPECIFIC_CFLAGS) $(I_CFLAGS) -c $< -o $@

# --- Test Workflow Targets ---
compile_tests: $(TEST_BIN)
	@echo ">>> [TESTS] Test runner $(TEST_BIN) is up to date or has been (re)built."
$(TEST_BIN): $(ALL_TEST_RUNNER_OBJS)
	@echo ">>> [TESTS] Linking test runner: $@"
	$(CLINKER) $(ALL_TEST_RUNNER_OBJS) -o $@ $(LDFLAGS) $(_COVERAGE_LINK_FLAGS) $(PETSC_LIB)
run_tests: compile_tests
	@echo ">>> [TESTS] Running tests..."
	./$(TEST_BIN)
test: run_tests

# Target to generate the COVERAGE report
coverage: build clean_coverage run_tests
coverage: build clean_coverage run_tests
	@echo ">>> [COVERAGE] Generating LCOV report..."
	lcov --capture --directory . --output-file build/coverage.raw.info --rc lcov_branch_coverage=1
	@echo ">>> [COVERAGE] Removing external libraries, tests, AND assembly.c from report..."
	lcov --remove build/coverage.raw.info '/usr/*' '/opt/*' '$(CURDIR)/tests/*' '$(CURDIR)/src/assembly.c' --output-file build/coverage.info --rc lcov_branch_coverage=1
	@echo ">>> [COVERAGE] Generating HTML report..."
	genhtml build/coverage.info --output-directory build/coverage-html --branch-coverage
	@echo ">>> [COVERAGE] HTML report generated at build/coverage-html/index.html"


# Clean coverage-related files AND test-specific object files
clean_coverage: build
	@echo ">>> [CLEAN] Removing coverage report files and test object files..."
	rm -f build/coverage.info
	rm -rf build/coverage-html
	find build/ -name '*.gcda' -delete
	find build/ -name '*.gcno' -delete
	if [ -d src ]; then find src/ -name '*.gcda' -delete; find src/ -name '*.gcno' -delete; fi
	if [ -d tests ]; then find tests/ -name '*.gcda' -delete; find tests/ -name '*.gcno' -delete; fi
	rm -f $(ALL_TEST_RUNNER_OBJS)

# === Documentation ===
DOXYFILE := Doxyfile
SPHINX_PY := python3 # Or just python if it's in your PATH and configured for Sphinx
SPHINX_API_SCRIPT_PATH := scripts/auto_doc/api_rst_generator.py
SPHINX_SOURCE_DIR := docs/source
SPHINX_BUILD_DIR := docs/build
SPHINX_BUILD_CMD := $(SPHINX_PY) -m sphinx # Recommended way to invoke Sphinx
SPHINX_OUT := $(SPHINX_BUILD_DIR)/html

# Target to generate all documentation
docs: clean_doc doxygen api_generator sphinx_html

# Generate Doxygen XML documentation
doxygen:
	@echo ">>> [DOCS] Generating Doxygen XML..."
	doxygen $(DOXYFILE)

# Generate structure for Sphinx
api_generator:
	@echo ">>> [DOCS] Running API generator (api_generator)..."
	$(SPHINX_PY) $(SPHINX_API_SCRIPT_PATH)

# Build HTML documentation with Sphinx
sphinx_html:
	@echo ">>> [DOCS] Building Sphinx HTML documentation..."
	LC_ALL=C.UTF-8 LANG=C.UTF-8 $(SPHINX_BUILD_CMD) -b html $(SPHINX_SOURCE_DIR) $(SPHINX_BUILD_DIR)/html
	@echo ">>> [DOCS] HTML documentation generated in $(SPHINX_BUILD_DIR)/html"

# Doc clean rule
clean_doc:
	@echo ">>> [CLEAN] Cleaning documentation..."
	rm -rf $(SPHINX_OUT)/* docs/doxygen/* docs/source/api/* docs/source/readme/*


# ==============================================================================
# === VALIDATION TEST                                                        ===
# ==============================================================================
# This block handles a full validation test, from data preprocessing to kernel
# execution and output verification.

# --- Validation Test Variables ---
# Scripts and Configuration Files
PREPROCESSING_SCRIPT      := data/preprocessing.py
OPTIONS_FILE              := data/options.txt

# Raw Input Data Files
RAW_MESH_FILE             := data/mesh.msh
RAW_RECEIVERS_FILE        := data/receivers.txt
RAW_SOURCES_FILE          := data/sources.txt

# Preprocessed Files (kernel inputs)
PREPROCESSED_MESH         := data/wham/wham_model.h5
PREPROCESSED_RECEIVERS    := data/wham/receivers.h5

# Final Output Files (kernel outputs)
VALIDATION_OUTPUT_1       := data/wham/responses_p1_src1.h5
VALIDATION_OUTPUT_2       := data/wham/responses_p1_src2.h5

# Kernel execution command (uses the main TARGET variable)
EXECUTE_KERNEL_CMD        := /usr/local/petsc/arch-linux-c-opt/bin/mpirun -n 1 $(TARGET) -options_file $(OPTIONS_FILE)


# --- Validation Test Rules ---

# Main target to run the entire validation test suite.
# Usage: `make validation_test`
.PHONY: validation_test
validation_test: all $(VALIDATION_OUTPUT_1) $(VALIDATION_OUTPUT_2)
	@echo
	@echo "============================================================"
	@echo ">>> [VALIDATION] SUCCESS: Test completed successfully."
	@echo ">>> [VALIDATION] Output files found:"
	@echo "    - $(VALIDATION_OUTPUT_1)"
	@echo "    - $(VALIDATION_OUTPUT_2)"
	@echo "============================================================"
	@echo


# Rule to generate the final output files.
# This rule is triggered if the output files are missing or if their
# dependencies (kernel, preprocessed files) are newer.
$(VALIDATION_OUTPUT_1) $(VALIDATION_OUTPUT_2): $(TARGET) $(PREPROCESSED_MESH) $(PREPROCESSED_RECEIVERS) $(OPTIONS_FILE) $(RAW_SOURCES_FILE)
	@echo ">>> [VALIDATION] Running PETGEM kernel to generate output..."
	$(EXECUTE_KERNEL_CMD)


# Rule to generate the preprocessed .h5 files.
# This rule is triggered if the .h5 files are missing or if the preprocessing
# script or raw data files are newer.
$(PREPROCESSED_MESH) $(PREPROCESSED_RECEIVERS): $(PREPROCESSING_SCRIPT) $(RAW_MESH_FILE) $(RAW_RECEIVERS_FILE)
	@echo ">>> [VALIDATION] Running preprocessing script..."
	python3 $(PREPROCESSING_SCRIPT)


# Rule to clean up all files generated by the validation test.
# Usage: `make clean_validation`
.PHONY: clean_validation
clean_validation:
	@echo ">>> [CLEAN] Removing validation test generated files..."
	rm -f $(PREPROCESSED_MESH) $(PREPROCESSED_RECEIVERS)
	rm -rf data/wham
	@echo ">>> [CLEAN] Validation files removed."


# === Cleaning ===
clean_all:
	@echo ">>> [CLEAN] Removing build artifacts..."
	rm -f $(OBJS) $(TARGET) $(TEST_BIN)
	$(MAKE) clean_doc
	@echo ">>> [CLEAN] Running PETSc clean..."
	$(MAKE) clean
	@echo ">>> [CLEAN] Removing coverage report files..."
	$(MAKE) clean_coverage
	@echo ">>> [CLEAN] Removing run files..."
	$(MAKE) clean_validation
	@echo ">>> [CLEAN] Removing build folder..."
	rm -rf ./build
