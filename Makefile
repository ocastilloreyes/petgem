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
.DEFAULT_GOAL := all

all: $(TARGET)
	@echo "$(C_LD)==>$(C_RESET) $(C_BOLD)PETGEM build complete$(C_RESET) -> $(TARGET)"

# -----------------------------------------------------------------------------
# Include PETSc-provided makefile configuration
# -----------------------------------------------------------------------------
# Fail with a clear message instead of the cryptic
#   "Makefile:NN: <dir>/lib/petsc/conf/variables: No such file or directory"
# that make would otherwise emit when PETSC_DIR is unset or wrong.
ifeq ($(wildcard $(PETSC_DIR)/lib/petsc/conf/variables),)
  $(error PETSC_DIR is unset or invalid (got '$(PETSC_DIR)'). \
    Point it at a PETSc install, e.g.  make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-...)
endif

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
    #           - informational IPO remark fired on the largest TUs
    #           (hvfem, hvfem_hierarchical, inversion).
    #   11076 = "To get full report use -qopt-report=4 -qopt-report-phase ipo"
    #           - companion to 11074; harmless suggestion.
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
# Console output styling
# -----------------------------------------------------------------------------
# ANSI colours for readable build output. Auto-disabled when stdout is not a
# terminal (CI logs, pipes) or when NO_COLOR is set (https://no-color.org).
# When disabled the C_* variables expand to empty, so messages degrade to
# plain text with no escape sequences.
#
# TTY detection uses MAKE_TERMOUT (GNU Make >= 4.1), which make sets to the
# terminal device when its OWN stdout is a TTY. A `$(shell test -t 1)` probe
# does NOT work here: inside $(shell ...) the subshell's stdout is a pipe
# back to make, so the test is always false even in an interactive terminal.
# On make < 4.1 MAKE_TERMOUT is unset, so colour stays off (safe fallback).
ifeq ($(NO_COLOR),)
  ifneq ($(MAKE_TERMOUT),)
    _ESC    := $(shell printf '\033')
    C_RESET := $(_ESC)[0m
    C_BOLD  := $(_ESC)[1m
    C_DIM   := $(_ESC)[2m
    C_CC    := $(_ESC)[36m
    C_LD    := $(_ESC)[32m
    C_CLEAN := $(_ESC)[33m
    C_DOC   := $(_ESC)[35m
  endif
endif

# Verbosity. `make V=1` echoes the full compiler/linker/tool command lines;
# the default hides them behind the concise [CC]/[LD]/[DOC] tags. $(Q) is
# prepended to the commands we normally want silenced.
V ?= 0
ifeq ($(V),1)
  Q :=
else
  Q := @
endif

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
SHARED_SRCS := src/common.c \
               src/io.c \
               src/grid.c \
               src/assembly.c \
               src/constants.c \
               src/fem.c \
               src/fe_nedelec.c \
               src/fe_nodal.c \
               src/solver.c \
               src/receiver_interp.c \
               src/postprocessing.c \
               src/mms.c

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

# Ordered-unique list of every object compiled in a full build, used to
# label compiles "[i/N] [CC] file.c". $(sort) is avoided because it would
# reorder alphabetically; this recursive uniq preserves first-occurrence
# order so the indices are monotonic for a serial `make`.
uniq = $(if $1,$(firstword $1) $(call uniq,$(filter-out $(firstword $1),$1)))
ALL_OBJS := $(call uniq,$(FM_OBJS) $(IM_OBJS) $(PT_OBJS))
NOBJS := $(words $(ALL_OBJS))

# -----------------------------------------------------------------------------
# Build rules for each kernel
# -----------------------------------------------------------------------------
# HDF5 flags. Raw HDF5 API (H5Fopen, H5Dread, H5Tcreate, …) is called
# from src/io.c which is linked into ALL three binaries, so fm.csem
# also needs -lhdf5 even though only loadObservedData uses the raw API.
HDF5_LIB := -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lhdf5

# One-time configuration banner, printed before any compilation. It is an
# order-only prerequisite of every object (below) so it always runs first
# without forcing rebuilds; .PHONY makes it fire once per invocation.
_preamble:
	@echo "$(C_BOLD)==> Building PETGEM$(C_RESET)"
	@echo "    compiler   : $(CC)"
	@echo "    options    : USE_INTEL=$(USE_INTEL)  USE_EXTRAE=$(USE_EXTRAE)  V=$(V)"
	@echo "    units      : $(NOBJS) objects -> $(words $(TARGET)) binaries"

$(FM_TARGET): $(FM_OBJS) | build
	@echo "$(C_LD)[LD]$(C_RESET) $@"
	$(Q)$(CLINKER) $(INTEL_DIAG) $^ -o $@ $(EXTRA_LDFLAGS) $(HDF5_LIB) $(PETSC_LIB)

$(IM_TARGET): $(IM_OBJS) | build
	@echo "$(C_LD)[LD]$(C_RESET) $@"
	$(Q)$(CLINKER) $(INTEL_DIAG) $^ -o $@ $(EXTRA_LDFLAGS) $(HDF5_LIB) $(PETSC_LIB)

# Unified binary: dispatches to runForward / runInverse based on
#   ./petgem modeling | inverse  (positional)  OR  -mode <modeling|inverse>
$(PT_TARGET): $(PT_OBJS) | build
	@echo "$(C_LD)[LD]$(C_RESET) $@"
	$(Q)$(CLINKER) $(INTEL_DIAG) $^ -o $@ $(EXTRA_LDFLAGS) $(HDF5_LIB) $(PETSC_LIB)

# Ensure object directory exists before compiling
$(OBJDIR):
	@mkdir -p $(OBJDIR)

# Per-object compile rules, generated one-per-object so each can show its
# own "[i/N]" progress index (baked in at parse time -> parallel-safe; no
# runtime counter state). We invoke $(CC) directly instead of PETSc's
# $(PETSC_COMPILE_SINGLE) because the latter injects a baked-in `-o<stem>.o`,
# which icc flags as a duplicate (#10122). PETSC_CC_INCLUDES is already part
# of BASE_CFLAGS, so this has everything it needs.
# $(1) = object path, $(2) = 1-based index baked into the progress label
define _emit_cc_rule
$(1): $(patsubst $(OBJDIR)/%.o,src/%.c,$(1)) | $(OBJDIR) _preamble
	@echo "$$(C_DIM)[$(2)/$(NOBJS)]$$(C_RESET) $$(C_CC)[CC]$$(C_RESET) $$<"
	$$(Q)$$(CC) $$(CFLAGS) $$(I_CFLAGS) -c $$< -o $$@
endef

__cc_ctr :=
$(foreach _o,$(ALL_OBJS),\
  $(eval __cc_ctr := $(__cc_ctr) x)\
  $(eval $(call _emit_cc_rule,$(_o),$(words $(__cc_ctr)))))

# Fallback pattern rule for any object not covered above (none today, but
# keeps the build robust if a new TU is added without regenerating). An
# explicit rule always wins over this pattern.
$(OBJDIR)/%.o: src/%.c | $(OBJDIR) _preamble
	@echo "$(C_CC)[CC]$(C_RESET) $<"
	$(Q)$(CC) $(CFLAGS) $(I_CFLAGS) -c $< -o $@

# Ensure build directory exists for binaries
build:
	@mkdir -p build

# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------
clean::         ## Remove object files and executables
	@echo "$(C_CLEAN)[CLEAN]$(C_RESET) removing build/ and object files"
	@rm -rf build $(OBJDIR)

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------
SPHINX_PY := python3
SPHINX_SOURCE_DIR := docs/source
SPHINX_BUILD_DIR := docs/build
SPHINX_BUILD_CMD := $(SPHINX_PY) -m sphinx
SPHINX_OUT := $(SPHINX_BUILD_DIR)/html
DOC_PREP_SCRIPT := scripts/auto_doc/prepare_docs.sh

docs: docs_prep sphinx_html                             ## Generate documentation (Doxygen + Sphinx HTML)
	@echo "$(C_DOC)==>$(C_RESET) $(C_BOLD)Documentation ready$(C_RESET): $(SPHINX_OUT)/index.html"

docs_prep:                                              ## Clean + Doxygen XML + Sphinx API .rst (shared with Read the Docs)
	@command -v doxygen >/dev/null 2>&1 || { echo "$(C_CLEAN)[DOC]$(C_RESET) ERROR: 'doxygen' not found on PATH (apt install doxygen)"; exit 1; }
	@echo "$(C_DOC)[DOC]$(C_RESET) preparing docs (clean + doxygen + api generator)"
	$(Q)bash $(DOC_PREP_SCRIPT)

sphinx_html:                                            ## Build HTML Sphinx documentation
	@$(SPHINX_PY) -c 'import sphinx' 2>/dev/null || { echo "$(C_CLEAN)[DOC]$(C_RESET) ERROR: Sphinx not installed (pip install -r docs/requirements.txt)"; exit 1; }
	@echo "$(C_DOC)[DOC]$(C_RESET) building Sphinx HTML"
	$(Q)LC_ALL=C.UTF-8 LANG=C.UTF-8 $(SPHINX_BUILD_CMD) -b html $(SPHINX_SOURCE_DIR) $(SPHINX_BUILD_DIR)/html

clean_doc:                                              ## Clean documentation output
	@echo "$(C_CLEAN)[CLEAN]$(C_RESET) removing generated documentation"
	@rm -rf $(SPHINX_OUT)/* docs/doxygen/* docs/source/api/* docs/source/readme/* docs/build/html

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:              ## Show this help message
	@echo "$(C_BOLD)PETGEM - make targets$(C_RESET)"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) \
        	| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(C_CC)%-12s$(C_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(C_BOLD)Binaries built by 'all'$(C_RESET) (suffixed .extrae when USE_EXTRAE=1):"
	@echo "  $(C_DIM)build/fm.csem$(C_RESET)   legacy forward kernel (single-purpose)"
	@echo "  $(C_DIM)build/im.csem$(C_RESET)   legacy inverse kernel (single-purpose)"
	@echo "  $(C_DIM)build/petgem$(C_RESET)    unified dispatcher: ./petgem modeling | inverse"
	@echo ""
	@echo "$(C_BOLD)Build options$(C_RESET) (set with 'make <target> OPTION=1'):"
	@echo "  USE_INTEL=1     force Intel MPI compiler flags (auto-detected from \$$(PCC))"
	@echo "  USE_EXTRAE=1    build with Extrae instrumentation"
	@echo "  NO_COLOR=1      disable coloured output"
	@echo ""
	@echo "$(C_BOLD)Documentation$(C_RESET):  make docs    (output: $(SPHINX_OUT)/index.html)"

# -----------------------------------------------------------------------------
# Phony targets - these never correspond to files, so always run regardless
# of any same-named file in the tree.
# -----------------------------------------------------------------------------
.PHONY: all _preamble clean help docs docs_prep sphinx_html clean_doc
