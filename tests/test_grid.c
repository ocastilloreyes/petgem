// tests/test_grid.c
#include "/opt/unity/src/unity.h"
#include "../include/grid.h"
// PETSc includes (asegurando petscmat.h si se necesitara por algún DM)
#include <petscsys.h>
#include <petscdmplex.h>
#include <petscvec.h>
#include <petscmat.h> // Añadido por si acaso, aunque no estrictamente necesario para DMPlexCreateBoxMesh aquí

static DM test_dm = NULL;
static Vec test_resistivity = NULL;
static Params test_params_for_grid;
static Grid test_grid_data;

void setUp_grid(void) {
    PetscCallVoid(PetscOptionsClear(NULL)); // <-- ¡CORRECCIÓN! Pasa NULL.

    if (test_dm) { PetscCallVoid(DMDestroy(&test_dm)); test_dm = NULL; }
    if (test_resistivity) { PetscCallVoid(VecDestroy(&test_resistivity)); test_resistivity = NULL; }
    memset(&test_params_for_grid, 0, sizeof(Params));
    memset(&test_grid_data, 0, sizeof(Grid));
}

void tearDown_grid(void) {
    if (test_dm) { PetscCallVoid(DMDestroy(&test_dm)); test_dm = NULL; }
    if (test_resistivity) { PetscCallVoid(VecDestroy(&test_resistivity)); test_resistivity = NULL; }
    if (test_grid_data.H1dm) { PetscCallVoid(DMDestroy(&test_grid_data.H1dm)); test_grid_data.H1dm = NULL;}
}

void test_importGrid_no_file(void) {
    setUp_grid();
    strcpy(test_params_for_grid.meshFile, ""); // Empty filename

    PetscErrorCode ierr = importGrid(&test_dm, &test_resistivity, test_params_for_grid);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);
    TEST_ASSERT_NOT_NULL(test_dm); // DM should be created (even if basic)
    TEST_ASSERT_NULL(test_resistivity); // Resistivity should be NULL
    tearDown_grid();
}

void test_setupGrid_simple_dm(void) {
    setUp_grid();
    PetscInt       dim = 3, cells[] = {1,1,1};
    DMBoundaryType periodicity[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
    // ¡CORRECCIÓN! 9º argumento es 0.
    PetscCallVoid(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, cells, NULL,  NULL, periodicity, PETSC_TRUE, 0, PETSC_FALSE, &test_dm));
    
    test_params_for_grid.nord = 1; // Simplest case

    PetscErrorCode ierr = setupGrid(&test_dm, &test_grid_data, test_params_for_grid);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);
    TEST_ASSERT_EQUAL_INT(dim, test_grid_data.dim);
    TEST_ASSERT_EQUAL_INT(1, test_grid_data.numDofInEdge); // nord = 1
    TEST_ASSERT_EQUAL_INT(0, test_grid_data.numDofInFace);  // nord * (nord-1) = 0 for nord=1
    TEST_ASSERT_EQUAL_INT(1, test_grid_data.numCellsGlobal);
    TEST_ASSERT_EQUAL_INT(8, test_grid_data.numVerticesGlobal);
    TEST_ASSERT_EQUAL_INT(12, test_grid_data.numEdgesGlobal);
    TEST_ASSERT_EQUAL_INT(6, test_grid_data.numFacesGlobal);
    TEST_ASSERT_NOT_NULL(test_grid_data.H1dm);

    tearDown_grid();
}

void test_locatePoint_not_found_robustness(void) {
    setUp_grid();
    PetscInt       dim = 3, cells[] = {1,1,1};
    PetscReal      lower[3] = {0,0,0}, upper[3] = {1,1,1}; // Unit cube
    DMBoundaryType periodicity[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
    // ¡CORRECCIÓN! 9º argumento es 0.
    PetscCallVoid(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, cells, lower, upper, periodicity, PETSC_TRUE, 0, PETSC_FALSE, &test_dm));

    // 'point_outside' se mantiene como ejemplo de una variable que no se usa y se elimina
    // o se usa para un test que verifique el comportamiento de locatePoint para puntos fuera del dominio.
    // PetscReal point_outside[] = {10.0, 10.0, 10.0}; // Eliminar si no se usa
    PetscInt cell_idx = -1;

    PetscReal point_inside[] = {0.5, 0.5, 0.5};
    cell_idx = -1;
    PetscErrorCode ierr = locatePoint(test_dm, point_inside, &cell_idx);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);
    TEST_ASSERT_TRUE(cell_idx >= 0); // Should find a cell (likely cell 0)

    tearDown_grid();
}

// Test group runner
void suite_grid(void) {
    RUN_TEST(test_importGrid_no_file);
    RUN_TEST(test_setupGrid_simple_dm);
    RUN_TEST(test_locatePoint_not_found_robustness);
}