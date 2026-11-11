/*
 * Filename: csem.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * CSEM physics: transmitters, the PEC grid, the dipole right-hand side and the
 * fields at the receivers.
 */

#ifndef CSEM_H
#define CSEM_H

#include <petsc.h>

#include "grid.h"
#include "io.h"

/** @brief simulation_type tag of fm.csem products. */
#define PETGEM_SIM_FM "fm.csem"

/**
 * @brief A single CSEM transmitter (electric dipole) record.
 */
typedef struct {
  PetscReal position[3];  /**< Transmitter position (x, y, z). */
  PetscReal current;      /**< Electric current. */
  PetscReal length;       /**< Dipole length. */
  PetscReal dipAngle;     /**< Dip angle. */
  PetscReal azimuthAngle; /**< Azimuth angle. */
} CsemSource;

/**
 * @brief A set of CSEM transmitters sharing one operating frequency.
 */
typedef struct {
  PetscReal   freq;        /**< Operating frequency (Hz). */
  PetscInt    numSources;  /**< Total number of transmitters. */
  CsemSource *sourceArray; /**< Array of `numSources` transmitter records. */
} CsemSourceSet;

/** @brief Columns of a /sources row (loadSourceRows). */
typedef enum {
  CSEM_SRC_FREQ,
  CSEM_SRC_X,
  CSEM_SRC_Y,
  CSEM_SRC_Z,
  CSEM_SRC_CURRENT,
  CSEM_SRC_LENGTH,
  CSEM_SRC_DIP,
  CSEM_SRC_AZIMUTH,
  CSEM_SRC_NUM_COLUMNS
} CsemSourceColumn;

/**
 * @brief Reads the /sources group of a bundle as a row table.
 *
 * Row i holds entry i in CSEM_SRC_* column order: frequency, position,
 * current, length, dip and azimuth.
 *
 * @param[in]  bundleFile  Path to the PETGEM HDF5 bundle.
 * @param[out] numRows     Number of entries.
 * @param[out] rows        numRows x CSEM_SRC_NUM_COLUMNS reals; free with PetscFree().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadSourceRows(const char *bundleFile, PetscInt *numRows, PetscReal **rows);

/**
 * @brief Loads the CSEM transmitter set from the bundle's /sources group.
 *
 * Reads on PETSC_COMM_SELF (each rank reads independently):
 *   - /sources/freq         per-entry frequency; freq[0] applies to the set
 *   - /sources/position     Vec of 3·N_src reals
 *   - /sources/current      Vec of N_src reals
 *   - /sources/length       Vec of N_src reals
 *   - /sources/dipAngle     Vec of N_src reals
 *   - /sources/azimuthAngle Vec of N_src reals
 *
 * @param[in]  params  Parameters; inputFile is read.
 * @param[out] sources    Transmitter set; sourceArray is allocated here.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadCsemSources(const PetgemParams *params, CsemSourceSet *sources);

/**
 * @brief Prints the "CSEM source(s)" section of the forward run report.
 *
 * @param[in] sources  Transmitter set loaded by loadCsemSources().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode logForwardSources(const CsemSourceSet *sources);

/**
 * @brief CSEM grid: setupNedelecGrid with PETGEM_BC_PEC.
 *
 * @param[in]     order   Basis order.
 * @param[in,out] dm      DMPlex mesh configured with the FE sections.
 * @param[out]    grid    Grid struct filled with mesh statistics and FemSpace.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode setupCsemGrid(PetscInt order, DM *dm, Grid *grid);

/**
 * @brief Assembles the CSEM right-hand side matrix.
 *
 * Produces B with one column per transmitter, with Dirichlet boundary DOFs
 * eliminated, ready for solveMaxwellSystem.
 *
 * @param[in]  sources  Transmitter set (one column of B per source).
 * @param[in]  dm       DMPlex mesh and H(curl) discretization.
 * @param[in]  grid     Finite-element grid descriptor.
 * @param[in]  constFactor  Fused-mode factor iωμ; ignored when Ms != NULL.
 * @param[out] B        Assembled right-hand side matrix.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode assembleCsemRhs(const CsemSourceSet sources, const DM dm, const Grid *grid, const PetscScalar constFactor, Mat *B);

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
 * viewer. `receivers` is the serial Vec (PETSC_COMM_SELF, length 3·N_recv)
 * returned by loadModelInputs.
 *
 * @param[in] params     Parameters (output paths).
 * @param[in] sources    Transmitter set (one solution column per source).
 * @param[in] dm         DMPlex mesh and H(curl) discretization.
 * @param[in] grid       Finite-element grid descriptor.
 * @param[in] receivers  Serial Vec of 3·N_recv receiver coordinates.
 * @param[in] X          Solution matrix, one column per source.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode computeCsemResponses(const PetgemParams *params, const CsemSourceSet sources, const DM dm, const Grid *grid, Vec receivers,
                                    const Mat X);

#endif /* CSEM_H */
