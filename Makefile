# ----------------------------------------------------------------------------- 
# PETGEM Makefile (dual-compiler support)
# ----------------------------------------------------------------------------- 
# This Makefile compiles and links the PETGEM kernel application with PETSc. 
# It supports optional integration with Extrae for tracing and ensures that 
# build artifacts are stored inside the "build" directory. 
# Supports PETSc default compiler and Intel MPI compiler (mpiicc). 
# ----------------------------------------------------------------------------- 

# ----------------------------------------------------------------------------- 
# Target executable
# ----------------------------------------------------------------------------- 
TARGET := build/csem_kernel
all: $(TARGET)		## Build the PETGEM kernels (default)

# ----------------------------------------------------------------------------- 
# Include PETSc-provided makefile configuration
# These bring in compiler settings, flags, and useful rules for PETSc builds. 
# ----------------------------------------------------------------------------- 
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# ----------------------------------------------------------------------------- 
# Optional Extrae instrumentation
# ----------------------------------------------------------------------------- 
USE_EXTRAE ?= 0

E_CFLAGS  := -DUSE_EXTRAE=$(USE_EXTRAE)
E_LDFLAGS :=

ifeq ($(USE_EXTRAE),1)
  E_CFLAGS  += -I$(EXTRAE_HOME)/include
  E_LDFLAGS += -L$(EXTRAE_HOME)/lib -lmpitrace
endif



# ----------------------------------------------------------------------------- 
# Include directory for PETGEM headers
# ----------------------------------------------------------------------------- 
I_CFLAGS := -Iinclude

# ----------------------------------------------------------------------------- 
# Choose compiler
# ----------------------------------------------------------------------------- 
USE_INTEL ?= 0
ifeq ($(USE_INTEL), 1)
    CC := mpiicc
    CFLAGS := -Iinclude ${PETSC_CC_INCLUDES} -O3 -g
else
    CC := ${PETSC_CC}
    CFLAGS := ${PETSC_CC_INCLUDES} -O3 -g
endif

# ----------------------------------------------------------------------------- 
# Source files for the PETGEM kernel
# ----------------------------------------------------------------------------- 
SRCS := src/csem_kernel.c \
        src/common.c \
        src/inputs.c \
        src/transmitter.c \
        src/grid.c \
        src/assembly.c \
        src/hvfem.c \
        src/solver.c \
        src/postprocessing.c
OBJS := $(SRCS:.c=.o)

# ----------------------------------------------------------------------------- 
# Build and cleaning rules
# ----------------------------------------------------------------------------- 

# Ensure build directory exists before linking
$(TARGET): | build

# Link all object files into the final executable
$(TARGET): $(OBJS)
	@echo "[LD] $@"
	@$(CLINKER) $^ -o $@ $(CFLAGS) $(E_LDFLAGS) $(PETSC_LIB)

# Compilation rule for each source file
%.o: %.c
	@echo "[CC] $<"
	@${PETSC_COMPILE_SINGLE} $(CFLAGS) $(I_CFLAGS) $(E_CFLAGS) $< -o $@

# Create build directory
build:
	@mkdir -p build

# Cleaning
clean::         ## Remove object files and executables
	@echo "[CLEAN]"
	@rm -rf build

# ----------------------------------------------------------------------------- 
# Documentation
# ----------------------------------------------------------------------------- 
DOXYFILE := Doxyfile
SPHINX_PY := python3      # Or just python if it's in your PATH and configured for Sphinx
SPHINX_API_SCRIPT_PATH := scripts/auto_doc/api_rst_generator.py
SPHINX_SOURCE_DIR := docs/source
SPHINX_BUILD_DIR := docs/build
SPHINX_BUILD_CMD := $(SPHINX_PY) -m sphinx # Recommended way to invoke Sphinx
SPHINX_OUT := $(SPHINX_BUILD_DIR)/html

# Target to generate all documentation
docs: clean_doc doxygen api_generator sphinx_html       ## Generate documentation

# Generate Doxygen XML documentation
doxygen:                                                ## Generate Doxygen XML documentation
	@echo ">>> [DOC] Generating Doxygen XML"
	doxygen $(DOXYFILE)

# Generate structure for Sphinx
api_generator:                                          ## Run Sphinx API generator
	@echo ">>> [DOC] Running API generator (api_generator)"
	$(SPHINX_PY) $(SPHINX_API_SCRIPT_PATH)

# Build HTML documentation with Sphinx
sphinx_html:                                            ## Build HTML Sphinx documentation 
	@echo ">>> [DOC] Building Sphinx HTML documentation..."
	LC_ALL=C.UTF-8 LANG=C.UTF-8 $(SPHINX_BUILD_CMD) -b html $(SPHINX_SOURCE_DIR) $(SPHINX_BUILD_DIR)/html
	@echo ">>> [DOCS] HTML documentation generated in $(SPHINX_BUILD_DIR)/html"

# Doc clean rule
clean_doc:                                              ## Clean documentation
	@echo ">>> [CLEAN] Cleaning documentation"
	rm -rf $(SPHINX_OUT)/* docs/doxygen/* docs/source/api/* docs/source/readme/* docs/build/html

# ----------------------------------------------------------------------------- 
# Help
# ----------------------------------------------------------------------------- 
help:              ## Show this help message
	@echo "Available make targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1;36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Optional build options (set with 'make <target> OPTION=1'):"
	@echo "  USE_INTEL=1     Use Intel MPI compiler (mpiicc) instead of PETSc default"
	@echo "  USE_EXTRAE=1    Enable Extrae instrumentation for tracing"
