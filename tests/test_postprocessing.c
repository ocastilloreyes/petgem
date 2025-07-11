// tests/test_postprocessing.c
#define _POSIX_C_SOURCE 200809L // Crucial for POSIX functions like stat, mkdir, S_ISDIR
#include "/opt/unity/src/unity.h"
#include "../include/postprocessing.h"
#include "../include/inputs.h"
#include "../include/grid.h"
#include "../include/source.h"
#include "../include/constants.h"
#include "../include/version.h" // For VERSION_MAJOR, etc.
#include "../include/common.h" // For createDirectory

// PETSc includes (ahora con petsc.h y petscviewerhdf5.h)
#include <petsc.h>
#include <petscviewerhdf5.h> // Necesario específicamente para HDF5 Viewer (no siempre en petsc.h)
#include <sys/stat.h>
#include <stdlib.h>


static DM       test_dm = NULL;
static Mat      test_X = NULL;
static Grid     test_grid;
static setSource test_sources;
static Params   test_params;
static Vec      temp_vec_for_dm_size = NULL; // Temporary vector to get DM size

void setUp_postprocessing(void) {
    PetscCallVoid(PetscOptionsClear(NULL)); // Correcto

    if (test_dm) PetscCallVoid(DMDestroy(&test_dm));
    if (test_X) PetscCallVoid(MatDestroy(&test_X));
    if (temp_vec_for_dm_size) PetscCallVoid(VecDestroy(&temp_vec_for_dm_size));

    memset(&test_params, 0, sizeof(Params));
    test_params.nord = 1;
    strcpy(test_params.mode, "CSEM");
    strcpy(test_params.receiversFile, "dummy_receivers.h5"); // Will be created by test
    strcpy(test_params.outputDirectory, "temp_output_postprocessing/"); // Needs to be created
    strcpy(test_params.outputFilename, "dummy_output");
    test_params.numMPITasks = 1;

    memset(&test_grid, 0, sizeof(Grid));
    test_grid.dim = 3;
    test_grid.numDofInCell = 6;
    test_grid.numCellsGlobal = 1; test_grid.numVerticesGlobal = 8; test_grid.numEdgesGlobal = 12;

    memset(&test_sources, 0, sizeof(setSource));
    test_sources.numSources = 1; test_sources.freq = 1.0;
    test_sources.sourceArray = (Source*)malloc(sizeof(Source) * test_sources.numSources);
    memset(test_sources.sourceArray, 0, sizeof(Source) * test_sources.numSources);
    test_sources.sourceArray[0].position[0] = 0.5;
    test_sources.sourceArray[0].position[1] = 0.5;
    test_sources.sourceArray[0].position[2] = 0.5;

    PetscInt dim = 3, cells[] = {1,1,1};
    PetscReal lower[3] = {0.0,0.0,0.0}, upper[3] = {1.0,1.0,1.0};
    DMBoundaryType periodicity[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
    PetscCallVoid(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, cells, lower, upper, periodicity, PETSC_TRUE, 0, PETSC_FALSE, &test_dm)); // Correcto
    PetscCallVoid(DMSetFromOptions(test_dm));
    PetscCallVoid(DMSetUp(test_dm));

    PetscInt M_dm;
    PetscCallVoid(DMGetLocalVector(test_dm, &temp_vec_for_dm_size));
    PetscCallVoid(VecGetSize(temp_vec_for_dm_size, &M_dm));
    PetscCallVoid(DMRestoreLocalVector(test_dm, &temp_vec_for_dm_size));
    temp_vec_for_dm_size = NULL;

    PetscCallVoid(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M_dm, test_sources.numSources, NULL, &test_X));
    PetscCallVoid(MatSetFromOptions(test_X));
    PetscCallVoid(MatAssemblyBegin(test_X, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(test_X, MAT_FINAL_ASSEMBLY));
    
    // ¡CORRECCIÓN AQUÍ! Reemplazamos MatSet
    PetscScalar *x_array_ptr;
    PetscCallVoid(MatDenseGetArray(test_X, &x_array_ptr));
    for (PetscInt i = 0; i < M_dm * test_sources.numSources; ++i) {
        x_array_ptr[i] = 1.0 + 1.0 * PETSC_i;
    }
    PetscCallVoid(MatDenseRestoreArray(test_X, &x_array_ptr));
    
    PetscCallVoid(createDirectory(test_params.outputDirectory));
}

void tearDown_postprocessing(void) {
    if (test_dm) PetscCallVoid(DMDestroy(&test_dm));
    if (test_X) PetscCallVoid(MatDestroy(&test_X));
    if (temp_vec_for_dm_size) PetscCallVoid(VecDestroy(&temp_vec_for_dm_size));
    if (test_sources.sourceArray) free(test_sources.sourceArray);

    char cmd[PETSC_MAX_PATH_LEN + 100];
    sprintf(cmd, "rm -rf %s", test_params.outputDirectory); system(cmd);
    sprintf(cmd, "rm -f %s", test_params.receiversFile); system(cmd);
}

void test_computeFields_basic_execution(void) {
    Vec mock_receivers;
    PetscCallVoid(VecCreate(PETSC_COMM_SELF, &mock_receivers));
    PetscCallVoid(VecSetSizes(mock_receivers, PETSC_DECIDE, 3));
    PetscCallVoid(VecSetFromOptions(mock_receivers));
    PetscScalar receiver_coords[] = {0.5, 0.5, 0.5};
    PetscInt  receiver_indices[] = {0, 1, 2};
    PetscCallVoid(VecSetValues(mock_receivers, 3, receiver_indices, receiver_coords, INSERT_VALUES));
    PetscCallVoid(VecAssemblyBegin(mock_receivers));
    PetscCallVoid(VecAssemblyEnd(mock_receivers));
    PetscCallVoid(PetscObjectSetName((PetscObject)mock_receivers,"receivers"));

    PetscViewer viewer_dummy;
    PetscCallVoid(PetscViewerHDF5Open(PETSC_COMM_SELF, test_params.receiversFile, FILE_MODE_WRITE, &viewer_dummy));
    PetscCallVoid(VecView(mock_receivers, viewer_dummy));
    PetscCallVoid(PetscViewerDestroy(&viewer_dummy));
    PetscCallVoid(VecDestroy(&mock_receivers));

    PetscErrorCode ierr = computeFields(test_dm, test_X, test_grid, test_sources, test_params);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);

    struct stat st;
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, stat(test_params.outputDirectory, &st), "Output directory should exist.");
    TEST_ASSERT_TRUE_MESSAGE(S_ISDIR(st.st_mode), "Output path should be a directory.");
}

void suite_postprocessing(void) {
    setUp_postprocessing();
    RUN_TEST(test_computeFields_basic_execution);
    tearDown_postprocessing();
}