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

/**
 * @brief Loads all CSEM inputs from the unified PETGEM HDF5 bundle.
 *
 * Opens fm_params->inputFile on PETSC_COMM_WORLD to load the DMPlex
 * topology, sections, and the combined model-data vector (split into the
 * per-cell conductivity and materials_id local Vecs). Opens the same file
 * again on PETSC_COMM_SELF (each rank reads independently) to load:
 *   - /nord                 single-element Vec, written into fm_params->nord
 *   - /receivers            Vec of 3·N_recv reals
 *   - /sources/frequency    single-frequency scalar
 *   - /sources/position     Vec of 3·N_src reals
 *   - /sources/current      Vec of N_src reals
 *   - /sources/length       Vec of N_src reals
 *   - /sources/dipAngle     Vec of N_src reals
 *   - /sources/azimuthAngle Vec of N_src reals
 *
 * Replaces the legacy importGrid + setupCsemSource + per-call
 * receivers-file open path with a single open of the bundle produced by
 * the Python preprocessor (utils/functions.py::writeBundle).
 *
 * @param[in,out] fm_params     Parameters; inputFile is read, nord is
 *                              written from the bundle's /nord dataset.
 * @param[out]    dm            Loaded DMPlex mesh.
 * @param[out]    conductivity  Per-cell conductivity Vec.
 * @param[out]    materialsID   Per-cell material-id Vec.
 * @param[out]    sources       Forward transmitter set; pass NULL to skip
 *                              (im.csem pulls multi-frequency sources via
 *                              setupInversionSources instead).
 * @param[out]    receivers     Serial Vec of 3·N_recv receiver reals; pass
 *                              NULL to skip.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadCsemInputs(fmParams       *fm_params,
                              DM             *dm,
                              Vec            *conductivity,
                              Vec            *materialsID,
                              CsemSourceSet  *sources,
                              Vec            *receivers);

#endif
