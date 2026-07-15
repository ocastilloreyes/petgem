/*
 * Filename: postprocessing.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Prototypes for the postprocessing functions used throughout PETGEM.
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "grid.h"
#include "io.h"
#include "transmitter.h"
#include <petsc.h>
#include <petscdmplex.h>

/**
 * @brief Computes electric and magnetic fields at receivers (forward kernel).
 *
 * Locates the receivers in the mesh, interpolates the H(curl) solution X to
 * obtain E (and H via H = curl(E)/(iωμ)) at each receiver, and writes a
 * SINGLE HDF5 response file containing every source. The file layout is:
 *
 *   /                              root attrs: petgem_version, input_filename,
 *                                              date, order, mpi_tasks,
 *                                              num_sources, frequency
 *   /sources/src{k}/               attrs: frequency, x_pos, y_pos, z_pos,
 *                                         current, length, dip_angle,
 *                                         azimuth_angle
 *   /sources/src{k}/fields/        Ex, Ey, Ez, Hx, Hy, Hz (PETSc Vec)
 *
 * The output filename is `{output_directory}/{output_filename}.h5`. All Vec
 * writes are collective on the kernel communicator through PETSc's HDF5
 * viewer (parallel HDF5 when PETSc is built against a parallel HDF5
 * library); no rank-0 gather happens. `receivers` is the serial Vec
 * (PETSC_COMM_SELF, length 3·N_recv) returned by loadCsemInputs - passed
 * through so postprocessing does not re-open the input HDF5.
 *
 * @param[in] params     Forward-modeling parameters (order, output paths).
 * @param[in] sources    Transmitter set (one solution column per source).
 * @param[in] dm         DMPlex mesh and H(curl) discretization.
 * @param[in] grid       Finite-element grid descriptor.
 * @param[in] receivers  Serial Vec of 3·N_recv receiver coordinates.
 * @param[in] X          Solution matrix, one column per source.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode computeFields(const petgemParams params, const CsemSourceSet sources,
                             const DM dm, const Grid grid,
                             Vec receivers, const Mat X);

#endif
