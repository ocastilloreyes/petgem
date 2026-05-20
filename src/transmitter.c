/*
 * Filename: transmitter.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Transmitter (CSEM source) types and helpers.
 */

/*
 * Notes:
 * The legacy text-format reader setupCsemSource() was replaced by
 * loadCsemInputs() (src/io.c), which pulls the single-frequency forward
 * sources out of the unified PETGEM input HDF5 (HDF5 group /sources).  This
 * translation unit now exists only to provide a place for transmitter-
 * related helpers if/when they are added.
 */

#include <petscsys.h>
#include "transmitter.h"
