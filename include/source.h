/*
  Filename: source.h
  Author: Octavio Castillo Reyes (UPC/BSC)
  Date: 2024-10-02
 
  Description:
  This file contains a collection of definitions for source functions that are used
  throughout the PETGEM project.

  Usage:
  Include this file in your source code to utilize the source functions. 
  For example:
  #include "source.h" 
*/

#ifndef SOURCE_H
#define SOURCE_H

#include "inputs.h"

/* =============================================================================
   Declaration of structures
   =============================================================================
*/
typedef struct {
   PetscReal position[3]; /* Source position (x, y, z) */    
   PetscReal current;     /* Electric current          */
   PetscReal length;      /* Dipole length             */
   PetscReal dip;         /* Dip                       */
   PetscReal azimuth;     /* Azimuth                   */
} Source;

typedef struct {
   PetscReal freq;        /* Frequency                 */
   PetscInt numSources;   /* Total number of sources */
   Source*  sourceArray;  /* Array of sources */
} setSource;


/* =============================================================================
   Declaration of functions
   =============================================================================
*/
PetscErrorCode setupSource(setSource* sources, Params params);

#endif 





