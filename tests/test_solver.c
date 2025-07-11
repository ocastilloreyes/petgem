// tests/test_solver.c
#include "/opt/unity/src/unity.h"
#include "../include/solver.h"
#include "../include/inputs.h"
#include "../include/constants.h"

// PETSc includes: ¡Ahora con petsc.h para ser exhaustivos!
#include <petsc.h>


static DM       test_dm = NULL;
static Mat      test_A = NULL;
static Mat      test_B = NULL;
static Mat      test_G = NULL;
static Mat      test_X = NULL;
static Params   test_params;
static Vec      temp_vec_for_dm_size = NULL;

void setUp_solver(void) {
    PetscCallVoid(PetscOptionsClear(NULL)); // Correcto

    if (test_dm) PetscCallVoid(DMDestroy(&test_dm));
    if (test_A) PetscCallVoid(MatDestroy(&test_A));
    if (test_B) PetscCallVoid(MatDestroy(&test_B));
    if (test_G) PetscCallVoid(MatDestroy(&test_G));
    if (test_X) PetscCallVoid(MatDestroy(&test_X));
    if (temp_vec_for_dm_size) PetscCallVoid(VecDestroy(&temp_vec_for_dm_size));

    memset(&test_params, 0, sizeof(Params));
    test_params.nord = 1;

    PetscInt dim = 3, cells[] = {1,1,1};
    DMBoundaryType periodicity[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
    PetscCallVoid(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, cells, NULL, NULL, periodicity, PETSC_TRUE, 0, PETSC_FALSE, &test_dm)); // Correcto
    PetscCallVoid(DMSetFromOptions(test_dm));
    PetscCallVoid(DMSetUp(test_dm));

    PetscInt global_size;
    PetscCallVoid(DMGetLocalVector(test_dm, &temp_vec_for_dm_size));
    PetscCallVoid(VecGetSize(temp_vec_for_dm_size, &global_size));
    PetscCallVoid(DMRestoreLocalVector(test_dm, &temp_vec_for_dm_size));
    temp_vec_for_dm_size = NULL;

    PetscCallVoid(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, global_size, global_size, 1, NULL, 0, NULL, &test_A));
    PetscCallVoid(MatSetFromOptions(test_A));
    PetscCallVoid(MatSetUp(test_A));
    PetscCallVoid(MatAssemblyBegin(test_A, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(test_A, MAT_FINAL_ASSEMBLY));
    
    // ¡CORRECCIÓN AQUÍ! Reemplazamos MatIdentity
    PetscCallVoid(MatZeroEntries(test_A)); // Pone todos los elementos a 0
    PetscCallVoid(MatShift(test_A, 1.0)); // Pone 1.0 en la diagonal (crea identidad)

    PetscInt num_sources = 1;
    PetscCallVoid(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, global_size, num_sources, NULL, &test_B));
    PetscCallVoid(MatSetFromOptions(test_B));
    PetscCallVoid(MatAssemblyBegin(test_B, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(test_B, MAT_FINAL_ASSEMBLY));
    
    // ¡CORRECCIÓN AQUÍ! Reemplazamos MatSet
    PetscScalar *b_array_ptr;
    PetscCallVoid(MatDenseGetArray(test_B, &b_array_ptr));
    for (PetscInt i = 0; i < global_size * num_sources; ++i) {
        b_array_ptr[i] = 1.0;
    }
    PetscCallVoid(MatDenseRestoreArray(test_B, &b_array_ptr));

    PetscInt H1_global_size = 8;
    PetscCallVoid(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, global_size, H1_global_size, 1, NULL, 0, NULL, &test_G));
    PetscCallVoid(MatSetFromOptions(test_G));
    PetscCallVoid(MatSetUp(test_G));
    PetscCallVoid(MatAssemblyBegin(test_G, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(test_G, MAT_FINAL_ASSEMBLY));
}

void tearDown_solver(void) {
    if (test_dm) PetscCallVoid(DMDestroy(&test_dm));
    if (test_A) PetscCallVoid(MatDestroy(&test_A));
    if (test_B) PetscCallVoid(MatDestroy(&test_B));
    if (test_G) PetscCallVoid(MatDestroy(&test_G));
    if (test_X) PetscCallVoid(MatDestroy(&test_X));
    if (temp_vec_for_dm_size) PetscCallVoid(VecDestroy(&temp_vec_for_dm_size));
}

void test_solveSystem_basic_execution(void) {
    PetscErrorCode ierr = solveSystem(test_dm, test_A, test_B, test_G, test_params, &test_X);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);
    TEST_ASSERT_NOT_NULL(test_X);

    PetscInt M_X, N_X;
    PetscCallVoid(MatGetSize(test_X, &M_X, &N_X));
    PetscInt M_B, N_B;
    PetscCallVoid(MatGetSize(test_B, &M_B, &N_B));

    TEST_ASSERT_EQUAL_INT(M_B, M_X);
    TEST_ASSERT_EQUAL_INT(N_B, N_X);

    PetscScalar *x_array;
    PetscInt num_local_rows_X;
    PetscCallVoid(MatGetLocalSize(test_X, &num_local_rows_X, NULL));
    PetscCallVoid(MatDenseGetArray(test_X, &x_array));
    for (PetscInt i = 0; i < num_local_rows_X * N_X; ++i) {
        TEST_ASSERT_EQUAL_DOUBLE(1.0, PetscRealPart(x_array[i]));
        TEST_ASSERT_EQUAL_DOUBLE(0.0, PetscImaginaryPart(x_array[i]));
    }
    PetscCallVoid(MatDenseRestoreArray(test_X, &x_array));
}

void suite_solver(void) {
    setUp_solver();
    RUN_TEST(test_solveSystem_basic_execution);
    tearDown_solver();
}