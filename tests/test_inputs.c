// tests/test_inputs.c
#define _POSIX_C_SOURCE 200809L // Para stat, S_ISDIR
#include "/opt/unity/src/unity.h"
#include <petscsys.h>    // Para PetscOptionsClearValue, PetscOptionsSetValue, PetscErrorCode, PETSC_SUCCESS, PetscOptionsClear
#include "../include/inputs.h" // Para Params, readUserParams
#include <string.h>    // For memset, strcpy
#include <sys/stat.h>  // For checking directory creation (stat, S_ISDIR)
#include <stdlib.h>    // For system()
#include <stdio.h>     // For remove()


static Params test_params;
static const char* temp_output_dir = "temp_test_output_inputs";


// Helper to clean up options and directories
void common_inputs_teardown() {
    PetscOptionsClearValue(NULL, "-mesh_filename");
    PetscOptionsClearValue(NULL, "-receivers_filename");
    PetscOptionsClearValue(NULL, "-output_dir");
    PetscOptionsClearValue(NULL, "-output_filename");
    PetscOptionsClearValue(NULL, "-nord");
    PetscOptionsClearValue(NULL, "-mode");
    PetscOptionsClearValue(NULL, "-source_filename");
    // Usar rm -rf para una limpieza más robusta
    char cmd[256];
    sprintf(cmd, "rm -rf %s", temp_output_dir);
    system(cmd);
}

void setUp_inputs(void) {
    PetscCallVoid(PetscOptionsClear(NULL)); // <-- ¡CORRECCIÓN! Pasa NULL.
    memset(&test_params, 0, sizeof(Params));
    common_inputs_teardown(); // Clean slate for options and directories
}

void tearDown_inputs(void) {
    common_inputs_teardown();
}

void test_readUserParams_valid(void) {
    setUp_inputs();
    PetscOptionsSetValue(NULL, "-mesh_filename", "dummy_mesh.h5");
    PetscOptionsSetValue(NULL, "-receivers_filename", "dummy_receivers.h5");
    PetscOptionsSetValue(NULL, "-output_dir", temp_output_dir);
    PetscOptionsSetValue(NULL, "-output_filename", "dummy_output");
    PetscOptionsSetValue(NULL, "-nord", "2");
    PetscOptionsSetValue(NULL, "-mode", "CSEM");
    PetscOptionsSetValue(NULL, "-source_filename", "dummy_source.txt");

    PetscErrorCode ierr = readUserParams(&test_params, 4);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);
    TEST_ASSERT_EQUAL_STRING("dummy_mesh.h5", test_params.meshFile);
    TEST_ASSERT_EQUAL_STRING("dummy_receivers.h5", test_params.receiversFile);
    TEST_ASSERT_EQUAL_STRING(temp_output_dir, test_params.outputDirectory);
    TEST_ASSERT_EQUAL_STRING("dummy_output", test_params.outputFilename);
    TEST_ASSERT_EQUAL_INT(2, test_params.nord);
    TEST_ASSERT_EQUAL_STRING("CSEM", test_params.mode);
    TEST_ASSERT_EQUAL_STRING("dummy_source.txt", test_params.sourceFilename);
    TEST_ASSERT_EQUAL_INT(4, test_params.numMPITasks);

    struct stat st;
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, stat(temp_output_dir, &st), "Output directory should be created.");
    TEST_ASSERT_TRUE_MESSAGE(S_ISDIR(st.st_mode), "Output path should be a directory.");
    tearDown_inputs();
}

void test_readUserParams_missing_mesh(void) {
    setUp_inputs();
    // Missing -mesh_filename
    PetscOptionsSetValue(NULL, "-receivers_filename", "dummy_receivers.h5");
    PetscOptionsSetValue(NULL, "-output_dir", temp_output_dir);
    PetscOptionsSetValue(NULL, "-output_filename", "dummy_output");
    PetscOptionsSetValue(NULL, "-nord", "2");
    PetscOptionsSetValue(NULL, "-mode", "CSEM");
    PetscOptionsSetValue(NULL, "-source_filename", "dummy_source.txt");

    PetscErrorCode ierr = readUserParams(&test_params, 1);
    TEST_ASSERT_NOT_EQUAL_INT(PETSC_SUCCESS, ierr);
    tearDown_inputs();
}

void test_readUserParams_invalid_nord(void) {
    setUp_inputs();
    PetscOptionsSetValue(NULL, "-mesh_filename", "dummy_mesh.h5");
    PetscOptionsSetValue(NULL, "-receivers_filename", "dummy_receivers.h5");
    PetscOptionsSetValue(NULL, "-output_dir", temp_output_dir);
    PetscOptionsSetValue(NULL, "-output_filename", "dummy_output");
    PetscOptionsSetValue(NULL, "-nord", "0"); // Invalid nord
    PetscOptionsSetValue(NULL, "-mode", "CSEM");
    PetscOptionsSetValue(NULL, "-source_filename", "dummy_source.txt");

    PetscErrorCode ierr = readUserParams(&test_params, 1);
    TEST_ASSERT_NOT_EQUAL_INT(PETSC_SUCCESS, ierr);

    PetscOptionsClearValue(NULL,"-nord"); // Clear previous setting before new attempt
    PetscOptionsSetValue(NULL, "-nord", "7"); // Another invalid nord
    ierr = readUserParams(&test_params, 1);
    TEST_ASSERT_NOT_EQUAL_INT(PETSC_SUCCESS, ierr);
    tearDown_inputs();
}

void test_readUserParams_invalid_mode(void) {
    setUp_inputs();
    PetscOptionsSetValue(NULL, "-mesh_filename", "dummy_mesh.h5");
    PetscOptionsSetValue(NULL, "-receivers_filename", "dummy_receivers.h5");
    PetscOptionsSetValue(NULL, "-output_dir", temp_output_dir);
    PetscOptionsSetValue(NULL, "-output_filename", "dummy_output");
    PetscOptionsSetValue(NULL, "-nord", "1");
    PetscOptionsSetValue(NULL, "-mode", "INVALID_MODE"); // Invalid mode
    PetscOptionsSetValue(NULL, "-source_filename", "dummy_source.txt");

    PetscErrorCode ierr = readUserParams(&test_params, 1);
    TEST_ASSERT_NOT_EQUAL_INT(PETSC_SUCCESS, ierr);
    tearDown_inputs();
}

// Test group runner
void suite_inputs(void) {
    setUp_inputs();
    RUN_TEST(test_readUserParams_valid);
    RUN_TEST(test_readUserParams_missing_mesh);
    RUN_TEST(test_readUserParams_invalid_nord);
    RUN_TEST(test_readUserParams_invalid_mode);
    tearDown_inputs();
}