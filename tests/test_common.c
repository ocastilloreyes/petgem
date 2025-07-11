// tests/test_common.c
#define _POSIX_C_SOURCE 200809L // Para stat, mkdir, S_ISDIR
#include "petsc.h" // Incluye petscsys.h y otros por defecto en la mayoría de configuraciones PETSc
#include "/opt/unity/src/unity.h"
#include "../include/common.h" // Para printHeader, printFooter, createDirectory
#include <sys/stat.h> // Para struct stat, stat, S_ISDIR
#include <stdio.h>    // Para FILE, fopen, fclose, remove
#include <stdlib.h>   // Para system()


// --- Funciones de configuración y limpieza de la suite ---
void setUp_common(void) {
    PetscCallVoid(PetscOptionsClear(NULL)); // <-- ¡CORRECCIÓN! Pasa NULL.
    // Limpiar cualquier residuo de directorios/archivos de tests anteriores
    system("rm -rf test_dir_new");
    system("rm -rf test_dir_existing");
    system("rm -f test_file_conflict_dir");
}

void tearDown_common(void) {
    // Asegurar que el entorno quede limpio después de la suite
    system("rm -rf test_dir_new");
    system("rm -rf test_dir_existing");
    system("rm -f test_file_conflict_dir");
}

// --- Individual Tests ---

// NOTA: Asegúrate de que tu printHeader y printFooter en src/common.c
// tienen los argumentos que pasas aquí (o no toman argumentos si son void).
// Basado en common.c que pasaste:
// printHeader(const char *version, const char *petsc_path, const char *petsc_arch)
// printFooter(const char *file_name, const double simulation_time, const double elapsed_time)
void test_printHeader_runs(void) {
    PetscErrorCode ierr = printHeader("2.0.0", "/usr/local/petsc", "arch-linux-c-opt"); // Pasa argumentos reales
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);
}

void test_printFooter_runs(void) {
    PetscErrorCode ierr = printFooter("output.h5", 10.5, 2.3); // Pasa argumentos reales
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);
}

void test_createDirectory_new(void) {
    const char* test_dir = "test_dir_new";
    PetscErrorCode ierr = createDirectory(test_dir);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);

    struct stat st;
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, stat(test_dir, &st), "Directory should exist after creation.");
    TEST_ASSERT_TRUE_MESSAGE(S_ISDIR(st.st_mode), "Path should be a directory.");
}

void test_createDirectory_existing(void) {
    const char* test_dir = "test_dir_existing";
    mkdir(test_dir, 0755); // Create it first using C's mkdir

    PetscErrorCode ierr = createDirectory(test_dir);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr); // Should succeed if already exists
}

void test_createDirectory_fileConflict(void) {
    const char* test_file_conflict = "test_file_conflict_dir";
    FILE* f = fopen(test_file_conflict, "w"); // Create a file with the same name
    if (f) fclose(f);
    else {
        TEST_FAIL_MESSAGE("Could not create conflicting file for test.");
        return;
    }

    PetscErrorCode ierr = createDirectory(test_file_conflict);
    TEST_ASSERT_NOT_EQUAL_INT(PETSC_SUCCESS, ierr); // Should fail
}

// Test group runner
void suite_common(void) {
    setUp_common();
    RUN_TEST(test_printHeader_runs);
    RUN_TEST(test_printFooter_runs);
    RUN_TEST(test_createDirectory_new);
    RUN_TEST(test_createDirectory_existing);
    RUN_TEST(test_createDirectory_fileConflict);
    tearDown_common();
}