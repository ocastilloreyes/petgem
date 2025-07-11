#include "/opt/unity/src/unity.h"
#include "petscsys.h" // For PetscInitialize/Finalize

// Forward declarations of test suite runners
void suite_common(void);
void suite_inputs(void);
void suite_source(void);
void suite_grid(void);
void suite_solver(void);
//void suite_assembly(void);
void suite_postprocessing(void);
void suite_hvfem(void);

// setUp and tearDown can be global if simple enough, or per-suite
void setUp(void) { }
void tearDown(void) { }

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));

    UnityBegin("PETGEM Tests");

    // Run test suites
    suite_common();
    suite_inputs();
    suite_source();
    suite_grid();
    suite_solver();
    //suite_assembly();
    suite_postprocessing();
    suite_hvfem();

    int failures = UnityEnd();

    PetscCall(PetscFinalize());
    return failures;
}