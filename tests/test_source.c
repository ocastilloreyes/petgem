// tests/test_source.c
#include "/opt/unity/src/unity.h"
#include <petscsys.h> // Para PetscOptionsClear(NULL)
#include "../include/transmitter.h" // Para setSource, Source, setupSource
#include <stdio.h>    // For FILE operations, remove
#include <string.h>   // For memset, strcpy
#include <stdlib.h>   // For system()


static setSource test_sources_data;
static Params test_params_for_source;
static const char* temp_source_file_csem = "temp_test_source_csem.txt";
static const char* temp_source_file_mt = "temp_test_source_mt.txt";
static const char* temp_source_file_bad_csem = "temp_test_source_bad_csem.txt";

void setUp_source(void) {
    PetscCallVoid(PetscOptionsClear(NULL)); // <-- ¡CORRECCIÓN! Pasa NULL.
    memset(&test_sources_data, 0, sizeof(setSource));
    memset(&test_params_for_source, 0, sizeof(Params));
    // Clean up any pre-existing temp files robustly
    system("rm -f temp_test_source_csem.txt");
    system("rm -f temp_test_source_mt.txt");
    system("rm -f temp_test_source_bad_csem.txt");
}

void tearDown_source(void) {
    if (test_sources_data.sourceArray) {
        PetscCallVoid(PetscFree(test_sources_data.sourceArray));
        test_sources_data.sourceArray = NULL;
    }
    system("rm -f temp_test_source_csem.txt");
    system("rm -f temp_test_source_mt.txt");
    system("rm -f temp_test_source_bad_csem.txt");
}

void create_csem_source_file(const char* filename, int num_sources, double freq, int valid_lines) {
    FILE* f = fopen(filename, "w");
    TEST_ASSERT_NOT_NULL_MESSAGE(f, "Failed to create temp CSEM source file.");
    fprintf(f, "%d\n", num_sources);
    fprintf(f, "%f\n", freq);
    for (int i = 0; i < num_sources; ++i) {
        if (i < valid_lines) {
            fprintf(f, "%f %f %f %f %f %f %f\n",
                    1.0*(i+1), 2.0*(i+1), 3.0*(i+1), // position
                    100.0, 10.0, 45.0, 90.0);    // current, length, dip, azimuth
        } else {
            fprintf(f, "bad data\n"); // To simulate bad format
        }
    }
    fclose(f);
}

void create_mt_source_file(const char* filename, int num_sources, double freq) {
    FILE* f = fopen(filename, "w");
    TEST_ASSERT_NOT_NULL_MESSAGE(f, "Failed to create temp MT source file.");
    fprintf(f, "%d\n", num_sources);
    fprintf(f, "%f\n", freq);
    fclose(f);
}

void test_setupSource_csem_valid(void) {
    setUp_source();
    int num_src = 2;
    double freq = 0.1;
    create_csem_source_file(temp_source_file_csem, num_src, freq, num_src);

    strcpy(test_params_for_source.mode, "CSEM");
    strcpy(test_params_for_source.sourceFilename, temp_source_file_csem);

    PetscErrorCode ierr = setupSource(&test_sources_data, test_params_for_source);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);
    TEST_ASSERT_EQUAL_INT(num_src, test_sources_data.numSources);
    TEST_ASSERT_EQUAL_DOUBLE(freq, test_sources_data.freq);
    TEST_ASSERT_NOT_NULL(test_sources_data.sourceArray);
    if (test_sources_data.sourceArray) { // Check array content
        TEST_ASSERT_EQUAL_DOUBLE(1.0, test_sources_data.sourceArray[0].position[0]);
        TEST_ASSERT_EQUAL_DOUBLE(100.0, test_sources_data.sourceArray[0].current);
        TEST_ASSERT_EQUAL_DOUBLE(4.0, test_sources_data.sourceArray[1].position[1]); // 2.0 * (1+1)
    }
    tearDown_source();
}

void test_setupSource_mt_valid(void) {
    setUp_source();
    int num_src = 1;
    double freq = 10.0;
    create_mt_source_file(temp_source_file_mt, num_src, freq);

    strcpy(test_params_for_source.mode, "MT");
    strcpy(test_params_for_source.sourceFilename, temp_source_file_mt);

    PetscErrorCode ierr = setupSource(&test_sources_data, test_params_for_source);
    TEST_ASSERT_EQUAL_INT(PETSC_SUCCESS, ierr);
    TEST_ASSERT_EQUAL_INT(num_src, test_sources_data.numSources);
    TEST_ASSERT_EQUAL_DOUBLE(freq, test_sources_data.freq);
    TEST_ASSERT_NOT_NULL(test_sources_data.sourceArray);
    if (test_sources_data.sourceArray) { // Check default MT values
        TEST_ASSERT_EQUAL_DOUBLE(0.0, test_sources_data.sourceArray[0].position[0]);
        TEST_ASSERT_EQUAL_DOUBLE(0.0, test_sources_data.sourceArray[0].current);
    }
    tearDown_source();
}

void test_setupSource_file_not_found(void) {
    setUp_source();
    strcpy(test_params_for_source.mode, "CSEM");
    strcpy(test_params_for_source.sourceFilename, "non_existent_source_file.txt");

    PetscErrorCode ierr = setupSource(&test_sources_data, test_params_for_source);
    TEST_ASSERT_NOT_EQUAL_INT(PETSC_SUCCESS, ierr);
    tearDown_source();
}

void test_setupSource_csem_bad_format(void) {
    setUp_source();
    int num_src = 2;
    double freq = 0.1;
    create_csem_source_file(temp_source_file_bad_csem, num_src, freq, 1); // Only 1 line is valid

    strcpy(test_params_for_source.mode, "CSEM");
    strcpy(test_params_for_source.sourceFilename, temp_source_file_bad_csem);

    PetscErrorCode ierr = setupSource(&test_sources_data, test_params_for_source);
    TEST_ASSERT_NOT_EQUAL_INT(PETSC_SUCCESS, ierr);
    tearDown_source();
}

// Test group runner
void suite_source(void) {
    setUp_source();
    RUN_TEST(test_setupSource_csem_valid);
    RUN_TEST(test_setupSource_mt_valid);
    RUN_TEST(test_setupSource_file_not_found);
    RUN_TEST(test_setupSource_csem_bad_format);
    tearDown_source();
}