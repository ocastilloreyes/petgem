/*
 * Filename: transmitter.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Type definitions for the CSEM transmitter (source) records used
 * throughout PETGEM.
 */

#ifndef TRANSMITTER_H
#define TRANSMITTER_H

/* Only PETSc scalar/int types are needed here (PetscReal, PetscInt); pull them
 * from <petsc.h> directly. Including io.h instead would be circular, since io.h
 * includes this header for CsemSourceSet. */
#include <petsc.h>

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
  PetscReal freq;          /**< Operating frequency (Hz). */
  PetscInt numSources;     /**< Total number of transmitters. */
  CsemSource* sourceArray; /**< Array of `numSources` transmitter records. */
} CsemSourceSet;

#endif
