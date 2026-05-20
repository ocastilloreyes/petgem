/*
 * Filename: io.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Public surface for the unified PETGEM input loader (loadCsemInputs).
 */

/*
 * Notes:
 * The loader replaces the legacy importGrid + setupCsemSource +
 * per-call receivers-file open path with a single open of the bundled
 * HDF5 file produced by the Python preprocessor
 * (utils/functions.py::writeBundle).
 */

#ifndef IO_H
#define IO_H

#include "inputs.h"
#include "transmitter.h"
#include <petsc.h>

/* Unified PETGEM input loader.
 *
 * Opens params->inputFile on PETSC_COMM_WORLD to load the DMPlex topology,
 * sections, and the combined model-data vector (which is split into the
 * per-cell conductivity and materials_id local Vecs). Opens the same file
 * a second time on PETSC_COMM_SELF (each rank reads independently) to load
 *   /nord                — single-element Vec; written into params->nord
 *                          (so the kernel no longer needs -nord in the
 *                          params file)
 *   /receivers           — Vec of 3·N_recv reals
 *   /sources/frequency   — single-frequency scalar
 *   /sources/position    — Vec of 3·N_src reals
 *   /sources/current     — Vec of N_src reals
 *   /sources/length      — Vec of N_src reals
 *   /sources/dipAngle    — Vec of N_src reals
 *   /sources/azimuthAngle — Vec of N_src reals
 *
 * `params` is in/out: inputFile is read, nord is written (from the bundle's
 * /nord dataset).  `sources` and `receivers` are optional: pass NULL for
 * either when the caller does not need it (e.g. im.csem passes NULL for
 * `sources` because it pulls multi-frequency sources from
 * params->sourceFilename via setupInversionSources). */
PetscErrorCode loadCsemInputs(csemParams     *params,
                              DM             *dm,
                              Vec            *conductivity,
                              Vec            *materialsID,
                              CsemSourceSet  *sources,
                              Vec            *receivers);

#endif
