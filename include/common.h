/*
 * Filename: common.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Prototypes for the common utility functions (printing helpers,
 * timers, …) used throughout PETGEM.
 */

#ifndef COMMON_H
#define COMMON_H

PetscErrorCode printHeader();

PetscErrorCode printFooter();

PetscErrorCode createDirectory(const char* path);

PetscErrorCode printTimers(const PetscLogDouble timers[]);

PetscErrorCode parseModeArg(const char *s, PetscInt *mode);

PetscErrorCode printUsage(const char *progname);

#endif
