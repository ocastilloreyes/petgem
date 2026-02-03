/*
 * Filename: transmitter.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-08-05
 *
 * Description:
 * This file contains functions for transmitter (CSEM or MT)
 * in a PETGEM simulation. It includes functions for parsing
 * source data based on user-provided parameters. The
 * functions in this file facilitate the setup and
 * configuration of the PETGEM code.
 *
 * Usage:
 * Include this file in your source code to utilize the
 * transmitter functions. For example: #include
 * "transmitter.h"
 */

/* C libraries */

/* PETSc libraries */
#include <petscsys.h>

/* PETGEM functions */
#include "transmitter.h"

/**
 * @brief Reads CSEM source parameters from a file.
 *
 * Opens the file specified in @p params.sourceFilename.
 * Reads the number of sources and the source frequency.
 * Allocates memory for @p sources->sourceArray.
 *
 * @param[out] sources Pointer to the setCsemSource struct
 * to be populated.
 * @param[in] params A CsemParams struct containing
 * simulation parameters, including source filename.
 * @return PetscErrorCode PETSC_SUCCESS on successful
 * reading and parsing. Returns error codes on file
 * open/read errors or format inconsistencies.
 */
PetscErrorCode setupCsemSource(const Params params, CsemSourceSet* sources) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt ret;
  PetscInt numSources = 0, idx = 0;
  char line[PETSC_MAX_PATH_LEN];
  long filePos;
  FILE* inputFile;

  /* Open source file */
  inputFile = fopen(params.sourceFilename, "r");
  PetscCheck(inputFile, PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "Exiting: Error opening source file %s.\n", params.sourceFilename);

  /* Read frequency from the first non-comment, non-empty
   * line */
  do {
    if (!fgets(line, sizeof(line), inputFile)) {
      PetscCheck(0, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ,
                 "Exiting: Error reading frequency from "
                 "source file %s.\n",
                 params.sourceFilename);
    }
  } while (line[0] == '#' || line[0] == '\n');

  /* Read source frequency */
  ret = sscanf(line, "%lf", &sources->freq);
  PetscCheck(ret == 1, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ,
             "Error parsing source frequency from source "
             "file %s.\n",
             params.sourceFilename);

  /* Count number of source lines */
  filePos = ftell(inputFile); /* save current position
                                 (after frequency) */
  while (fgets(line, sizeof(line), inputFile)) {
    if (line[0] == '#' || line[0] == '\n')
      continue; /* Skip comment/empty */
    numSources++;
  }
  sources->numSources = numSources;

  /* Allocate memory for sources */
  PetscCall(PetscMalloc1(sources->numSources, &sources->sourceArray));

  /* Rewind to start reading source lines */
  fseek(inputFile, filePos, SEEK_SET);

  /* Parsing source data */
  while (fgets(line, sizeof(line), inputFile) && idx < sources->numSources) {
    if (line[0] == '#' || line[0] == '\n')
      continue;

    ret = sscanf(line, "%lf %lf %lf %lf %lf %lf %lf", &sources->sourceArray[idx].position[0], &sources->sourceArray[idx].position[1],
                 &sources->sourceArray[idx].position[2], &sources->sourceArray[idx].current, &sources->sourceArray[idx].length,
                 &sources->sourceArray[idx].dipAngle, &sources->sourceArray[idx].azimuthAngle);
    PetscCheck(ret == 7, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ,
               "Exiting: Error parsing CSEM source data at "
               "line %d.",
               idx + 2);

    idx++;
  }

  /* Close file */
  fclose(inputFile);

  /* Print source data */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nCSEM source data:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Freq (Hz)         = %g\n", sources->freq));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Number of sources = %" PetscInt_FMT "\n", sources->numSources));
  for (PetscInt i = 0; i < sources->numSources; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Data for source %" PetscInt_FMT ":\n", i + 1));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Current         = %g\n", sources->sourceArray[i].current));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Length          = %g\n", sources->sourceArray[i].length));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Dip             = %g\n", sources->sourceArray[i].dipAngle));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Azimuth         = %g\n", sources->sourceArray[i].azimuthAngle));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Position (xyz)  = [%g, %g, %g]\n", sources->sourceArray[i].position[0],
                          sources->sourceArray[i].position[1], sources->sourceArray[i].position[2]));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
