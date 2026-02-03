/*
 * Filename: solver.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-09-04
 *
 * Description:
 * This file contains functions for handling input data and
 * user parameters in a PETGEM simulation. It includes
 * functions for parsing input data, and processing
 * user-provided parameters. The functions in this file
 * facilitate the setup and configuration of the PETGEM
 * code.
 *
 * Usage:
 * Include this file in your source code to utilize the
 * input functions. For example: #include "inputs.h"
 */

/* C libraries */
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

/* PETSc libraries */
#include <petsc.h>
#include <petscsys.h>

/* PETGEM functions */
#include "common.h"
#include "inputs.h"

/**
 * @brief Reads, parses, and validates user-provided CSEM simulation parameters from PETSc options.
 *
 * @param[in] size The total number of MPI tasks (MPI_Comm_size).
 * @param[out] params Pointer to a `csemParams` struct to populate with validated parameters.
 *
 * @return PetscErrorCode
 *   - PETSC_SUCCESS on successful parsing and validation.
 *   - PETSC_ERR_ARG_NULL or other PETSc error codes if mandatory parameters are missing or invalid.
 *
 * @details
 * This function reads required CSEM simulation parameters from the PETSc options database
 * (command-line options or options file) and populates the `params` structure. It also
 * performs basic validation of mandatory parameters and value ranges. The following
 * options are read:
 *
 * - `-mesh_filename`        : Path to the mesh file (mandatory, HDF5 format).
 * - `-receivers_filename`   : Path to the receivers file (mandatory, HDF5 format).
 * - `-output_dir`           : Directory where output files will be written (mandatory).
 * - `-output_filename`      : Base name for output files (mandatory).
 * - `-nord`                 : Finite element basis order (integer, 1 ≤ nord ≤ 3, mandatory).
 * - `-source_filename`      : Path to the source definition file (mandatory).
 *
 * The function performs the following:
 * 1. Checks that each mandatory parameter is provided.
 * 2. Validates that `nord` is within the allowed range (1–3).
 * 3. Copies all string parameters into the `params` struct safely using `PetscStrncpy`.
 * 4. Stores the number of MPI tasks in `params->numMPITasks`.
 * 5. Creates the output directory if it does not exist by calling `createDirectory`.
 *
 * @note
 * - If any mandatory parameter is missing or invalid, the function terminates with an
 *   informative PETSc error message.
 * - This function is intended to be called **before any simulation setup** to ensure all
 *   required parameters are present and valid.
 */
PetscErrorCode readCsemParams(const PetscMPIInt size, csemParams* params) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  char meshFilename[PETSC_MAX_PATH_LEN];
  char receiversFilename[PETSC_MAX_PATH_LEN];
  char outputDir[PETSC_MAX_PATH_LEN];
  char outputFilename[PETSC_MAX_PATH_LEN];
  char sourceFilename[PETSC_MAX_PATH_LEN];
  PetscBool meshFilenameIsPresent, receiversFilenameIsPresent, nordIsPresent, sourceFilenameIsPresent;
  PetscBool outputDirIsPresent, outputFilenameIsPresent;
  PetscInt nord; /* Basis order = 1, 2, 3 */

  /* Read mesh filename (hdf5 format) */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-mesh_filename", meshFilename, sizeof(meshFilename), &meshFilenameIsPresent));
  PetscCheck(meshFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Exiting: Mesh file missing. Mandatory "
             "parameter required for simulation.\n");
  PetscCall(PetscStrncpy(params->meshFile, meshFilename, sizeof(params->meshFile)));

  /* Read receivers filename (hdf5 format) */
  PetscCall(
      PetscOptionsGetString(NULL, NULL, "-receivers_filename", receiversFilename, sizeof(receiversFilename), &receiversFilenameIsPresent));
  PetscCheck(receiversFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Exiting: Receivers file missing. Mandatory "
             "parameter required for simulation.\n");
  PetscCall(PetscStrncpy(params->receiversFile, receiversFilename, sizeof(params->receiversFile)));

  /* Read output directory */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-output_dir", outputDir, sizeof(outputDir), &outputDirIsPresent));
  PetscCheck(outputDirIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Exiting: Output directory missing. Mandatory "
             "parameter required for simulation.\n");
  PetscCall(PetscStrncpy(params->outputDirectory, outputDir, sizeof(params->outputDirectory)));

  /* Read output filename */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-output_filename", outputFilename, sizeof(outputFilename), &outputFilenameIsPresent));
  PetscCheck(outputFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Exiting: Output filename missing. Mandatory "
             "parameter required for simulation.\n");
  PetscCall(PetscStrncpy(params->outputFilename, outputFilename, sizeof(params->outputFilename)));

  /* Read basis order (nord = 1, 2, 3) */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nord", &nord, &nordIsPresent));
  PetscCheck(nordIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Exiting: Nord parameter missing. Mandatory "
             "parameter required for simulation.\n");
  /* Check is a valid basis order */
  nordIsPresent = (nord >= 1 && nord <= 3) ? PETSC_TRUE : PETSC_FALSE;
  PetscCheck(nordIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Exiting: Nord parameter out of valid range "
             "(nord = 1, 2, 3).\n");
  params->nord = nord;

  /* Read source filename */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-source_filename", sourceFilename, sizeof(sourceFilename), &sourceFilenameIsPresent));
  PetscCheck(sourceFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Exiting: Source filename missing. Mandatory "
             "parameter required for simulation.\n");
  PetscCall(PetscStrncpy(params->sourceFilename, sourceFilename, sizeof(params->sourceFilename)));

  /* Number of MPI tasks */
  params->numMPITasks = size;

  /* Create output directory */
  createDirectory(outputDir);

  PetscFunctionReturn(PETSC_SUCCESS);
}
