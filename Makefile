# -----------------------------------------------------------------------------
# Select binary names, flags, and object directory based on USE_EXTRAE
# -----------------------------------------------------------------------------
USE_EXTRAE ?= 0
ifeq ($(USE_EXTRAE), 1)
    EXTRA_CFLAGS := -I$(EXTRAE_HOME)/include -DUSE_EXTRAE
    EXTRA_LDFLAGS := -L$(EXTRAE_HOME)/lib -lmpitrace
    OBJDIR := build/extrae
else
    EXTRA_CFLAGS :=
    EXTRA_LDFLAGS :=
    OBJDIR := build/noextrae
endif

# Target names
FM_TARGET := build/fm.csem
IM_TARGET := build/im.csem
PT_TARGET := build/petgem
ifeq ($(USE_EXTRAE),1)
    FM_TARGET := build/fm.csem.extrae
    IM_TARGET := build/im.csem.extrae
    PT_TARGET := build/petgem.extrae
endif

TARGET := $(FM_TARGET) $(IM_TARGET) $(PT_TARGET)

# -----------------------------------------------------------------------------
# Build all targets by default
# -----------------------------------------------------------------------------
all: $(TARGET)

# -----------------------------------------------------------------------------
# Include PETSc-provided makefile configuration
# -----------------------------------------------------------------------------
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# -----------------------------------------------------------------------------
# Choose compiler and compilation flags
# -----------------------------------------------------------------------------
# Always use PETSc's MPI-aware C compiler ($(PCC) is populated by the
# included variables file above; on a gcc PETSc it's typically mpicc, on
# an Intel PETSc it's mpiicc or mpiicx).
CC := $(PCC)

# Auto-detect Intel compiler from $(PCC) so the user doesn't have to pass
# USE_INTEL=1 manually on Intel-PETSc systems (e.g. MareNostrum 5).
# `make USE_INTEL=0` still forces gcc-style flags if needed.
#
# Care: `mpicc` (gcc) contains the substring "icc"; a plain findstring
# would false-positive.  We match (a) the *basename* starting with
# icc/icx/icpc/icpx, or (b) the full path containing mpiicc/mpiicx.
PCC_NAME := $(notdir $(PCC))
INTEL_DETECTED := $(filter icc% icx% icpc% icpx%,$(PCC_NAME))$(findstring mpiicc,$(PCC))$(findstring mpiicx,$(PCC))
USE_INTEL ?= $(if $(INTEL_DETECTED),1,0)

ifeq ($(USE_INTEL), 1)
    # Silence icc diagnostics that are noise rather than signal:
    #   10441 = "ICC is deprecated" remark
    #   10148 = "-Wpedantic not supported" (PETSc's $(CLINKER) inherits
    #           -Wpedantic from PETSc's own configure-time flags; we
    #           cannot easily strip it, so we silence the warning).
    #   11074 = "Inlining inhibited by limit max-size / max-total-size"
    #           — informational IPO remark fired on the largest TUs
    #           (hvfem, hvfem_hierarchical, inversion).
    #   11076 = "To get full report use -qopt-report=4 -qopt-report-phase ipo"
    #           — companion to 11074; harmless suggestion.
    # INTEL_DIAG is propagated to both CFLAGS and the link command.
    INTEL_DIAG := -diag-disable=10441,10148,11074,11076
    BASE_CFLAGS := $(PETSC_CC_INCLUDES) -O3 -g $(INTEL_DIAG)
    WARN_CFLAGS := -Wall -Wextra \
                   -Wunused-variable \
                   -Wunused-function \
                   -Wunused-parameter
else
    INTEL_DIAG :=
    BASE_CFLAGS := $(PETSC_CC_INCLUDES) -O3 -g
    WARN_CFLAGS := -Wall -Wextra -Wpedantic \
                   -Wunused-variable \
                   -Wunused-function \
                   -Wunused-parameter \
                   -Wunused-but-set-variable
endif

# Final CFLAGS: base flags + optional Extrae
CFLAGS := $(BASE_CFLAGS) $(WARN_CFLAGS) $(EXTRA_CFLAGS)

# Include directory for PETGEM headers
I_CFLAGS := -Iinclude

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
SHARED_SRCS := src/common.c \
               src/io.c \
               src/transmitter.c \
               src/grid.c \
               src/assembly.c \
               src/constants.c \
               src/hvfem.c \
               src/hvfem_hierarchical.c \
               src/solver.c \
               src/receiver_interp.c \
               src/postprocessing.c

# Kernel-specific sources
# fm_csem.c and im_csem.c expose runForward / runInverse; the *_main.c
# files are 3-line wrappers providing main() for the legacy binaries.
# inversion.c was split into 3 inversion-only TUs (inversion + smoother +
# lbfgs); the I/O half lives in shared src/io.c (merged with inputs.c).
# They share private prototypes via include/inversion_internal.h.
INV_SRCS := src/inversion.c src/inversion_smoother.c src/lbfgs.c

FM_SRCS := src/fm_csem.c src/fm_csem_main.c $(SHARED_SRCS)
IM_SRCS := src/im_csem.c src/im_csem_main.c $(INV_SRCS) $(SHARED_SRCS)
# Unified petgem binary: dispatcher main + both kernel cores + inversion + shared.
PT_SRCS := src/petgem.c src/fm_csem.c src/im_csem.c $(INV_SRCS) $(SHARED_SRCS)

# Object files
FM_OBJS := $(patsubst src/%.c,$(OBJDIR)/%.o,$(FM_SRCS))
IM_OBJS := $(patsubst src/%.c,$(OBJDIR)/%.o,$(IM_SRCS))
PT_OBJS := $(patsubst src/%.c,$(OBJDIR)/%.o,$(PT_SRCS))

# -----------------------------------------------------------------------------
# Build rules for each kernel
# -----------------------------------------------------------------------------
# HDF5 flags. Raw HDF5 API (H5Fopen, H5Dread, H5Tcreate, …) is called
# from src/io.c which is linked into ALL three binaries, so fm.csem
# also needs -lhdf5 even though only loadObservedData uses the raw API.
HDF5_LIB := -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lhdf5

$(FM_TARGET): $(FM_OBJS) | build
	@echo "[LD] $@"
	@$(CLINKER) $(INTEL_DIAG) $^ -o $@ $(EXTRA_LDFLAGS) $(HDF5_LIB) $(PETSC_LIB)

$(IM_TARGET): $(IM_OBJS) | build
	@echo "[LD] $@"
	@$(CLINKER) $(INTEL_DIAG) $^ -o $@ $(EXTRA_LDFLAGS) $(HDF5_LIB) $(PETSC_LIB)

# Unified binary: dispatches to runForward / runInverse based on
#   ./petgem modeling | inverse  (positional)  OR  -mode <modeling|inverse>
$(PT_TARGET): $(PT_OBJS) | build
	@echo "[LD] $@"
	@$(CLINKER) $(INTEL_DIAG) $^ -o $@ $(EXTRA_LDFLAGS) $(HDF5_LIB) $(PETSC_LIB)

# Ensure object directory exists before compiling
$(OBJDIR):
	@mkdir -p $(OBJDIR)

# Compilation rule for each source file.
# We invoke $(CC) directly instead of PETSc's $(PETSC_COMPILE_SINGLE) because
# the latter injects a baked-in `-o<stem>.o`, which icc flags as a duplicate
# (#10122 "overriding -ofoo.o with -obuild/.../foo.o").  PETSC_CC_INCLUDES is
# already part of BASE_CFLAGS, so this rule has everything it needs.
$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	@echo "[CC] $<"
	@$(CC) $(CFLAGS) $(I_CFLAGS) -c $< -o $@

# Ensure build directory exists for binaries
build:
	@mkdir -p build

# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------
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
	@echo "Binaries built by 'all':"
	@echo "  build/fm.csem   legacy forward kernel (single-purpose)"
	@echo "  build/im.csem   legacy inverse kernel (single-purpose)"
	@echo "  build/petgem    unified dispatcher: ./petgem modeling | inverse"
	@echo ""
	@echo "Optional build options (set with 'make <target> OPTION=1'):"
	@echo "  USE_INTEL=1     Use Intel MPI compiler (mpiicc) instead of PETSc default"
	@echo "  USE_EXTRAE=1    Build binary with Extrae support"
