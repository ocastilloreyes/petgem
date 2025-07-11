/*
 * Filename: receivers.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2024-06-04
 *
 * Description:
 * This file contains a collection of definitions for receivers functions that are used
 * throughout the PETGEM project.
 *
 * List of definitions:
 * -  
 *
 * Usage:
 * Include this file in your source code to utilize the common functions. 
 *
 * For example:
 * #include "receivers.h"
 * 
*/

#ifndef SOURCE_H
#define SOURCE_H

// =============================================================================
// Declaration of structures
// =============================================================================



// =============================================================================
// Declaration of functions
// =============================================================================
PetscErrorCode setupReceivers(petgemSource *source, int mode);

#endif