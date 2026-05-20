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

#include "inputs.h"

typedef struct {
  PetscReal position[3];  /* Transmitter position (x, y, z) */
  PetscReal current;      /* Electric current          */
  PetscReal length;       /* Dipole length             */
  PetscReal dipAngle;     /* Dip angle                 */
  PetscReal azimuthAngle; /* Azimuth angle             */
} CsemSource;

typedef struct {
  PetscReal freq;          /* Frequency                 */
  PetscInt numSources;     /* Total number of transmitters */
  CsemSource* sourceArray; /* Array of sources */
} CsemSourceSet;

#endif
