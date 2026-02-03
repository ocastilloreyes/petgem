# -----------------------------------------------------------------------------
# Select binary name, flags, and object directory based on USE_EXTRAE
# -----------------------------------------------------------------------------
USE_EXTRAE ?= 0
ifeq ($(USE_EXTRAE), 1)
    TARGET := build/fm.csem.extrae
    OBJDIR := build/extrae
    EXTRA_CFLAGS := -I$(EXTRAE_HOME)/include -DUSE_EXTRAE
    EXTRA_LDFLAGS := -L$(EXTRAE_HOME)/lib -lmpitrace
else
    TARGET := build/fm.csem
    OBJDIR := build/noextrae
    EXTRA_CFLAGS :=
    EXTRA_LDFLAGS :=
endif

# Build the PETGEM kernels
all: $(TARGET)

# -----------------------------------------------------------------------------
# Include PETSc-provided makefile configuration
# -----------------------------------------------------------------------------
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# -----------------------------------------------------------------------------
# Choose compiler and compilation flags
# -----------------------------------------------------------------------------
USE_INTEL ?= 0
ifeq ($(USE_INTEL), 1)
    CC := mpiicc
    BASE_CFLAGS := ${PETSC_CC_INCLUDES} -O3 -g
else
    CC := ${PETSC_CC}
    BASE_CFLAGS := ${PETSC_CC_INCLUDES} -O3 -g
endif

# -----------------------------------------------------------------------------
# Warning flags (unused code detection)
# -----------------------------------------------------------------------------
WARN_CFLAGS := -Wall -Wextra -Wpedantic \
               -Wunused-variable \
               -Wunused-function \
               -Wunused-parameter \
               -Wunused-but-set-variable

# Final CFLAGS: base flags + optional Extrae
CFLAGS := $(BASE_CFLAGS) $(WARN_CFLAGS) $(EXTRA_CFLAGS)


# Include directory for PETGEM headers
I_CFLAGS := -Iinclude

# -----------------------------------------------------------------------------
# Source files for the PETGEM kernel
# -----------------------------------------------------------------------------
SRCS := src/fm_csem.c \
        src/common.c \
        src/inputs.c \
        src/transmitter.c \
        src/grid.c \
        src/assembly.c \
        src/constants.c \
        src/hvfem.c \
        src/solver.c \
        src/postprocessing.c

OBJS := $(patsubst src/%.c,$(OBJDIR)/%.o,$(SRCS))

# -----------------------------------------------------------------------------
# Build and cleaning rules
# -----------------------------------------------------------------------------

# Ensure object directory exists before compiling
$(OBJDIR):
	@mkdir -p $(OBJDIR)

# Link all object files into the final executable
$(TARGET): $(OBJS) | build
	@echo "[LD] $@"
	@$(CLINKER) $^ -o $@ $(EXTRA_LDFLAGS) $(PETSC_LIB)

# Compilation rule for each source file
$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	@echo "[CC] $<"
	@${PETSC_COMPILE_SINGLE} $(CFLAGS) $(I_CFLAGS) -c $< -o $@

# Ensure build directory exists for binaries
build:
	@mkdir -p build

# Cleaning
clean::         ## Remove object files and executables
	@echo "[CLEAN]"
	@rm -rf build $(OBJDIR)

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------
DOXYFILE := Doxyfile
SPHINX_PY := python3
SPHINX_API_SCRIPT_PATH := scripts/auto_doc/api_rst_generator.py
SPHINX_SOURCE_DIR := docs/source
SPHINX_BUILD_DIR := docs/build
SPHINX_BUILD_CMD := $(SPHINX_PY) -m sphinx
SPHINX_OUT := $(SPHINX_BUILD_DIR)/html

docs: clean_doc doxygen api_generator sphinx_html       ## Generate documentation

doxygen:                                                ## Generate Doxygen XML documentation
	@echo ">>> [DOC] Generating Doxygen XML"
	doxygen $(DOXYFILE)

api_generator:                                          ## Run Sphinx API generator
	@echo ">>> [DOC] Running API generator (api_generator)"
	$(SPHINX_PY) $(SPHINX_API_SCRIPT_PATH)

sphinx_html:                                            ## Build HTML Sphinx documentation 
	@echo ">>> [DOC] Building Sphinx HTML documentation..."
	LC_ALL=C.UTF-8 LANG=C.UTF-8 $(SPHINX_BUILD_CMD) -b html $(SPHINX_SOURCE_DIR) $(SPHINX_BUILD_DIR)/html
	@echo ">>> [DOCS] HTML documentation generated in $(SPHINX_BUILD_DIR)/html"

clean_doc:                                              ## Clean documentation
	@echo ">>> [CLEAN] Cleaning documentation"
	rm -rf $(SPHINX_OUT)/* docs/doxygen/* docs/source/api/* docs/source/readme/* docs/build/html

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:              ## Show this help message
	@echo "Available make targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1;36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Optional build options (set with 'make <target> OPTION=1'):"
	@echo "  USE_INTEL=1     Use Intel MPI compiler (mpiicc) instead of PETSc default"
	@echo "  USE_EXTRAE=1    Build binary with Extrae support
