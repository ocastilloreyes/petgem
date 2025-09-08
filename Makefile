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
all: $(TARGET)

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
ifeq ($(USE_EXTRAE), 1)
    E_CFLAGS  := -I$(EXTRAE_HOME)/include -DUSE_EXTRAE
    E_LDFLAGS := -L$(EXTRAE_HOME)/lib -lmpitrace
else
    E_CFLAGS  :=
    E_LDFLAGS :=
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
# Build rules
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

# ----------------------------------------------------------------------------- 
# Cleaning
# ----------------------------------------------------------------------------- 
clean::
	@echo "[CLEAN]"
	@rm -f $(OBJS) $(TARGET)
