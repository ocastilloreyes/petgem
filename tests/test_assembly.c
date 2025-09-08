// tests/test_assembly.c
#include "/opt/unity/src/unity.h"
#include "../include/assembly.h"
#include "../include/inputs.h"
#include "../include/grid.h"
#include "../include/transmitter.h"
#include "../include/constants.h"
#include <petscdmplex.h> // For DMPlexCreateBoxMesh, DMPlexCreateSection etc.
#include <petscsection.h> // For PetscSection related functions

// Static variables for the test suite
static DM test_dm_assembly = NULL;
static DM test_dm_resistivity_assembly = NULL; // For the resistivity Vec
static Vec test_resistivity_vec_assembly = NULL;
static Grid test_grid_assembly;
static setSource test_sources_assembly;
static Params test_params_assembly;
static Mat test_A_assembly = NULL, test_B_assembly = NULL, test_G_assembly = NULL;

void setUp_assembly(void) {
    PetscCallVoid(PetscOptionsClear(NULL));

    // Clean up previous state
    if (test_A_assembly) { MatDestroy(&test_A_assembly); test_A_assembly = NULL; }
    if (test_B_assembly) { MatDestroy(&test_B_assembly); test_B_assembly = NULL; }
    if (test_G_assembly) { MatDestroy(&test_G_assembly); test_G_assembly = NULL; }
    if (test_resistivity_vec_assembly) { VecDestroy(&test_resistivity_vec_assembly); test_resistivity_vec_assembly = NULL; }
    if (test_dm_resistivity_assembly) { DMDestroy(&test_dm_resistivity_assembly); test_dm_resistivity_assembly = NULL; }
    if (test_grid_assembly.H1dm) { DMDestroy(&test_grid_assembly.H1dm); test_grid_assembly.H1dm = NULL; }
    if (test_dm_assembly) { DMDestroy(&test_dm_assembly); test_dm_assembly = NULL; }
    if (test_sources_assembly.sourceArray) { free(test_sources_assembly.sourceArray); test_sources_assembly.sourceArray = NULL; }


    // 1. Params
    memset(&test_params_assembly, 0, sizeof(Params));
    test_params_assembly.nord = 1; // For simplicity, nord = 1
    strcpy(test_params_assembly.mode, "CSEM");
    test_params_assembly.numMPITasks = 1; // Assuming serial run for test
    strcpy(test_params_assembly.meshFile, "dummy_mesh_for_assembly_test.h5"); // Required by setupGrid for printing

    // 2. setSource
    memset(&test_sources_assembly, 0, sizeof(setSource));
    test_sources_assembly.numSources = 1;
    test_sources_assembly.freq = 1.0;
    test_sources_assembly.sourceArray = (Source*)malloc(sizeof(Source) * test_sources_assembly.numSources);
    TEST_ASSERT_NOT_NULL(test_sources_assembly.sourceArray);
    memset(test_sources_assembly.sourceArray, 0, sizeof(Source) * test_sources_assembly.numSources);
    test_sources_assembly.sourceArray[0].position[0] = 0.5;
    test_sources_assembly.sourceArray[0].position[1] = 0.5;
    test_sources_assembly.sourceArray[0].position[2] = 0.5; // Center of unit cube
    test_sources_assembly.sourceArray[0].current = 1.0;
    test_sources_assembly.sourceArray[0].length = 1.0;
    test_sources_assembly.sourceArray[0].dip = 0.0;
    test_sources_assembly.sourceArray[0].azimuth = 0.0;

    // 3. DM for H(curl) space
    PetscInt dim = 3, cells_per_dim[] = {1,1,1}; 
    PetscReal lower[] = {0,0,0}, upper[] = {1,1,1};
    DMBoundaryType periodicity[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
    PetscCallVoid(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, cells_per_dim, lower, upper, periodicity, PETSC_TRUE, 0, PETSC_FALSE, &test_dm_assembly));
    PetscCallVoid(DMSetFromOptions(test_dm_assembly));
    PetscCallVoid(DMSetUp(test_dm_assembly));

    // 4. Grid structure (calls setupGrid)
    memset(&test_grid_assembly, 0, sizeof(Grid));
    PetscCallVoid(setupGrid(&test_dm_assembly, &test_grid_assembly, test_params_assembly));

    // 5. Resistivity Vec
    PetscSection resSection;
    PetscCallVoid(DMClone(test_dm_assembly, &test_dm_resistivity_assembly));
    PetscInt numCompRes[] = {3};
    PetscInt numDofResCell[] = {0,0,0,3}; 
    PetscCallVoid(DMSetNumFields(test_dm_resistivity_assembly, 1)); 
    PetscCallVoid(DMPlexCreateSection(test_dm_resistivity_assembly, NULL, numCompRes, numDofResCell, 0, NULL, NULL, NULL, NULL, &resSection));
    PetscCallVoid(DMSetLocalSection(test_dm_resistivity_assembly, resSection));
    PetscCallVoid(PetscSectionDestroy(&resSection));

    PetscCallVoid(DMCreateLocalVector(test_dm_resistivity_assembly, &test_resistivity_vec_assembly));
    PetscScalar *res_array;
    PetscCallVoid(VecGetArray(test_resistivity_vec_assembly, &res_array));
    PetscInt cellStart, cellEnd;
    PetscCallVoid(DMPlexGetHeightStratum(test_dm_resistivity_assembly, 0, &cellStart, &cellEnd));
    for (PetscInt c = cellStart; c < cellEnd; ++c) {
        PetscInt off;
        PetscSection cell_section_for_res = NULL; // Use a different variable name
        PetscCallVoid(DMGetLocalSection(test_dm_resistivity_assembly, &cell_section_for_res));
        if (cell_section_for_res) { // Check if section exists
            PetscCallVoid(PetscSectionGetOffset(cell_section_for_res, c, &off));
            if (off >=0) { 
              res_array[off + 0] = 10.0; 
              res_array[off + 1] = 10.0; 
              res_array[off + 2] = 10.0; 
            }
        }
    }
    PetscCallVoid(VecRestoreArray(test_resistivity_vec_assembly, &res_array));
}

void tearDown_assembly(void) {
    if (test_A_assembly) { MatDestroy(&test_A_assembly); test_A_assembly = NULL; }
    if (test_B_assembly) { MatDestroy(&test_B_assembly); test_B_assembly = NULL; }
    if (test_G_assembly) { MatDestroy(&test_G_assembly); test_G_assembly = NULL; }
    if (test_resistivity_vec_assembly) { VecDestroy(&test_resistivity_vec_assembly); test_resistivity_vec_assembly = NULL; }
    if (test_dm_resistivity_assembly) { DMDestroy(&test_dm_resistivity_assembly); test_dm_resistivity_assembly = NULL; }
    if (test_grid_assembly.H1dm) { DMDestroy(&test_grid_assembly.H1dm); test_grid_assembly.H1dm = NULL; }
    if (test_dm_assembly) { DMDestroy(&test_dm_assembly); test_dm_assembly = NULL; }
    if (test_sources_assembly.sourceArray) { free(test_sources_assembly.sourceArray); test_sources_assembly.sourceArray = NULL; }
}

void test_assembleSystem_csem_nord1_execution_and_dims(void) {
    setUp_assembly(); 

    PetscCallVoid(PetscPrintf(PETSC_COMM_WORLD, "test_assembleSystem: Before assembleSystem call. grid.numCellsGlobal = %" PetscInt_FMT ", grid.numVerticesGlobal = %" PetscInt_FMT ", grid.numEdgesGlobal = %" PetscInt_FMT ", grid.numFacesGlobal = %" PetscInt_FMT "\n",
        test_grid_assembly.numCellsGlobal, test_grid_assembly.numVerticesGlobal, test_grid_assembly.numEdgesGlobal, test_grid_assembly.numFacesGlobal));

    PetscSection section_check_hcurl; 
    PetscInt num_fields_check_hcurl = 0, p_start_check_hcurl = 0, p_end_check_hcurl = 0, num_section_dof_hcurl = 0;
    PetscCallVoid(DMGetLocalSection(test_dm_assembly, &section_check_hcurl)); 
    if (section_check_hcurl) { 
        PetscCallVoid(PetscSectionGetNumFields(section_check_hcurl, &num_fields_check_hcurl));
        PetscCallVoid(PetscSectionGetChart(section_check_hcurl, &p_start_check_hcurl, &p_end_check_hcurl));
        for (PetscInt p = p_start_check_hcurl; p < p_end_check_hcurl; ++p) {
            PetscInt dof;
            PetscCallVoid(PetscSectionGetDof(section_check_hcurl, p, &dof));
            num_section_dof_hcurl += dof;
        }
        PetscCallVoid(PetscPrintf(PETSC_COMM_WORLD, "test_assembleSystem: DM H(curl) section has %" PetscInt_FMT " fields, chart [%" PetscInt_FMT ", %" PetscInt_FMT "), total DOFs in H(curl) section (local) = %" PetscInt_FMT "\n", num_fields_check_hcurl, p_start_check_hcurl, p_end_check_hcurl, num_section_dof_hcurl));
        PetscInt local_storage_size_hcurl;
        PetscCallVoid(PetscSectionGetStorageSize(section_check_hcurl, &local_storage_size_hcurl));
        PetscCallVoid(PetscPrintf(PETSC_COMM_WORLD, "test_assembleSystem: DM H(curl) section local storage size = %" PetscInt_FMT "\n", local_storage_size_hcurl));
    } else {
        PetscCallVoid(PetscPrintf(PETSC_COMM_WORLD, "test_assembleSystem: DM H(curl) section is NULL.\n"));
    }

    PetscSection section_check_h1; 
    PetscInt num_fields_check_h1 = 0, p_start_check_h1 = 0, p_end_check_h1 = 0, num_section_dof_h1 = 0;
    if (test_grid_assembly.H1dm) { 
        PetscCallVoid(DMGetLocalSection(test_grid_assembly.H1dm, &section_check_h1));
        if (section_check_h1) {
            PetscCallVoid(PetscSectionGetNumFields(section_check_h1, &num_fields_check_h1));
            PetscCallVoid(PetscSectionGetChart(section_check_h1, &p_start_check_h1, &p_end_check_h1));
            for (PetscInt p = p_start_check_h1; p < p_end_check_h1; ++p) {
                PetscInt dof;
                PetscCallVoid(PetscSectionGetDof(section_check_h1, p, &dof));
                num_section_dof_h1 += dof;
            }
            PetscCallVoid(PetscPrintf(PETSC_COMM_WORLD, "test_assembleSystem: DM H1 section has %" PetscInt_FMT " fields, chart [%" PetscInt_FMT ", %" PetscInt_FMT "), total DOFs in H1 section (local) = %" PetscInt_FMT "\n", num_fields_check_h1, p_start_check_h1, p_end_check_h1, num_section_dof_h1));
            PetscInt local_storage_size_h1;
            PetscCallVoid(PetscSectionGetStorageSize(section_check_h1, &local_storage_size_h1));
            PetscCallVoid(PetscPrintf(PETSC_COMM_WORLD, "test_assembleSystem: DM H1 section local storage size = %" PetscInt_FMT "\n", local_storage_size_h1));
        } else {
             PetscCallVoid(PetscPrintf(PETSC_COMM_WORLD, "test_assembleSystem: DM H1 section is NULL.\n"));
        }
    } else {
        PetscCallVoid(PetscPrintf(PETSC_COMM_WORLD, "test_assembleSystem: grid.H1dm is NULL.\n"));
    }

    PetscErrorCode ierr = assembleSystem(test_dm_assembly, test_resistivity_vec_assembly, test_grid_assembly,
                                         test_sources_assembly, test_params_assembly,
                                         &test_A_assembly, &test_B_assembly, &test_G_assembly);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);

    TEST_ASSERT_NOT_NULL(test_A_assembly);
    TEST_ASSERT_NOT_NULL(test_B_assembly);
    TEST_ASSERT_NOT_NULL(test_G_assembly);

    PetscInt M_A, N_A_cols, M_B, N_B_cols, M_G, N_G_cols;
    PetscCallVoid(MatGetSize(test_A_assembly, &M_A, &N_A_cols));
    PetscCallVoid(MatGetSize(test_B_assembly, &M_B, &N_B_cols));
    PetscCallVoid(MatGetSize(test_G_assembly, &M_G, &N_G_cols));

    PetscCallVoid(PetscPrintf(PETSC_COMM_WORLD, "test_assembleSystem: After assembleSystem. A size: %" PetscInt_FMT "x%" PetscInt_FMT ", B size: %" PetscInt_FMT "x%" PetscInt_FMT ", G size: %" PetscInt_FMT "x%" PetscInt_FMT "\n", M_A, N_A_cols, M_B, N_B_cols, M_G, N_G_cols));

    PetscInt expected_hcurl_dofs = test_grid_assembly.numEdgesGlobal;
    PetscInt expected_h1_dofs = test_grid_assembly.numVerticesGlobal;

    TEST_ASSERT_TRUE_MESSAGE(expected_hcurl_dofs > 0 || test_grid_assembly.numCellsGlobal == 0, "Expected H(curl) DOFs is zero for a non-empty mesh, check DM/grid setup.");
    TEST_ASSERT_TRUE_MESSAGE(expected_h1_dofs > 0 || test_grid_assembly.numCellsGlobal == 0, "Expected H1 DOFs is zero for a non-empty mesh, check DM/grid setup.");

    TEST_ASSERT_EQUAL_INT_MESSAGE(expected_hcurl_dofs, M_A, "Matrix A row count mismatch.");
    TEST_ASSERT_EQUAL_INT_MESSAGE(expected_hcurl_dofs, N_A_cols, "Matrix A column count mismatch.");

    TEST_ASSERT_EQUAL_INT_MESSAGE(expected_hcurl_dofs, M_B, "Matrix B row count mismatch.");
    TEST_ASSERT_EQUAL_INT_MESSAGE(test_sources_assembly.numSources, N_B_cols, "Matrix B column count mismatch.");

    TEST_ASSERT_EQUAL_INT_MESSAGE(expected_hcurl_dofs, M_G, "Matrix G row count mismatch.");
    TEST_ASSERT_EQUAL_INT_MESSAGE(expected_h1_dofs, N_G_cols, "Matrix G column count mismatch.");

    // This check depends on the specific tetrahedralization of a 1x1x1 cube
    // For PETSc's default DMPlexCreateBoxMesh with interpolate=PETSC_TRUE, a 1x1x1 box often becomes 6 tetrahedra.
    // Such a mesh has 8 vertices, 19 edges, 18 faces, 6 cells.
    if (test_grid_assembly.numCellsGlobal == 6) {
         TEST_ASSERT_EQUAL_INT(8, test_grid_assembly.numVerticesGlobal);
         TEST_ASSERT_EQUAL_INT(19, test_grid_assembly.numEdgesGlobal);
         // You could also check numFacesGlobal if needed, should be 18 for this specific case
    } else if (test_grid_assembly.numCellsGlobal == 1 && test_grid_assembly.numEdgesGlobal == 12) {
        // This case would imply the mesh remained a single hexahedron, which is not the expectation with interpolate=PETSC_TRUE
        // but could be a fallback if interpolation fails or if setupGrid misinterprets the DM.
         TEST_ASSERT_EQUAL_INT(8, test_grid_assembly.numVerticesGlobal);
    } else {
        // Print a warning if the cell count is unexpected, as other counts might also be off.
        PetscCallVoid(PetscPrintf(PETSC_COMM_WORLD, "WARNING: Unexpected number of global cells (%" PetscInt_FMT ") for a 1x1x1 interpolated box. DOF counts might be affected.\n", test_grid_assembly.numCellsGlobal));
    }

    PetscReal norm_A;
    if (M_A > 0 && N_A_cols > 0) { 
        PetscCallVoid(MatNorm(test_A_assembly, NORM_FROBENIUS, &norm_A));
        TEST_ASSERT_TRUE_MESSAGE(norm_A > PETSC_SMALL, "Matrix A norm is too small, possibly all zeros.");
    } else {
        TEST_FAIL_MESSAGE("Matrix A is 0x0, cannot compute norm.");
    }

    PetscReal norm_B;
    if (M_B > 0 && N_B_cols > 0) {
        PetscCallVoid(MatNorm(test_B_assembly, NORM_FROBENIUS, &norm_B));
        TEST_ASSERT_TRUE_MESSAGE(norm_B > PETSC_SMALL || test_sources_assembly.numSources == 0, "Matrix B norm is too small for CSEM case.");
    } else if (test_sources_assembly.numSources > 0 && M_A > 0) { 
        TEST_FAIL_MESSAGE("Matrix B is 0-dimensional in rows or cols, but sources exist and A has DOFs.");
    }

    PetscReal norm_G;
    if (M_G > 0 && N_G_cols > 0) {
        PetscCallVoid(MatNorm(test_G_assembly, NORM_FROBENIUS, &norm_G));
        TEST_ASSERT_TRUE_MESSAGE(norm_G > PETSC_SMALL, "Matrix G norm is too small, possibly all zeros.");
    } else {
         TEST_FAIL_MESSAGE("Matrix G is 0-dimensional, cannot compute norm.");
    }

    tearDown_assembly();
}

void suite_assembly(void) {
    RUN_TEST(test_assembleSystem_csem_nord1_execution_and_dims);
}