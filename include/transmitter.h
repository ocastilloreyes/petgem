/*
  Filename: transmitter.h
  Author: Octavio Castillo Reyes (UPC/BSC)
  Date: 2025-09-05
 
  Description:
  This file contains a collection of definitions for transmitter functions that are used
  throughout the PETGEM project.

  Usage:
  Include this file in your source code to utilize the transmitter functions. 
  For example:
  #include "transmitter.h" 
*/

#ifndef TRANSMITTER_H
#define TRANSMITTER_H

#include "inputs.h"

typedef struct {
   PetscReal position[3]; /* Transmitter position (x, y, z) */    
   PetscReal current;     /* Electric current          */
   PetscReal length;      /* Dipole length             */
   PetscReal dip;         /* Dip                       */
   PetscReal azimuth;     /* Azimuth                   */
} CsemSource;

typedef struct {
   PetscReal freq;        /* Frequency                 */
   PetscInt numSources;   /* Total number of transmitters */
   CsemSource*  sourceArray;  /* Array of sources */
} setCsemSource;


PetscErrorCode setupSource(setCsemSource* sources, Params params);

#endif 





