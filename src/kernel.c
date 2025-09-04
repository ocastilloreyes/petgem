static char help[] = "PETGEM kernel for 3D CSEM modeling using high-order vector finite elements.\n\
  Command line usage:\n\
    mpirun -n <np> ./kernel -options_file <file.txt>\n"

/* C libraries */ 
#include <stdio.h>
#include <stdlib.h>

/* PETSc functions */   
#include <petsc.h>  
#include <petscsys.h>  
#include <petscdmplex.h>

/* PETGEM functions */ 
#include "version.h"
#include "constants.h"
#include "common.h"
#include "inputs.h"  
#include "source.h" 
#include "grid.h" 
#include "assembly.h"
#include "solver.h"  
#include "postprocessing.h"    

/* Extrae library for performance analysis */
#ifdef USE_EXTRAE
#include "extrae_user_events.h"  
#endif

/**
 * @brief Main execution routine for CSEM kernel.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return int 0 on success, non-zero on failure.
 * @details Initializes PETSc, parses command-line arguments (including `--version`).
 *          Prints the header. Reads user parameters, sets up sources, imports the grid
 *          and resistivity, sets up the DM sections and grid structure, assembles the
 *          linear system (A, B, G), solves the system (AX=B) using KSP, performs
 *          post-processing (computes fields at receivers), prints the footer,
 *          and finalizes PETSc, freeing allocated memory. Includes Extrae instrumentation hooks if compiled with USE_EXTRAE.
  */
int main(int argc, char **argv)
{

    /* Check if the --version option is provided */
    if (argc > 1 && strcmp(argv[1], "--version") == 0) {
        printf("PETGEM version %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
        return 0;
    }

    /* Variables declaration */
    PetscMPIInt rank, size;
    DM      dm;
    Vec     resistivity;
    Mat     A, B, X;
    Mat     G = NULL;
    Params  params;  
    Grid    grid; 
    setSource sources = {0, 0, NULL};

    /* PETSC initialization */
    PetscFunctionBeginUser;
    #ifdef USE_EXTRAE
    Extrae_event (1000, 1);
    #endif
    PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    #ifdef USE_EXTRAE
    Extrae_event (1000, 0);
    #endif

    /* Print PETGEM header */
    #ifdef USE_EXTRAE
    Extrae_event (1000, 2);
    #endif
    PetscCall(printHeader());
    #ifdef USE_EXTRAE
    Extrae_event (1000, 0);
    #endif

    /* Parse user parameters */
    #ifdef USE_EXTRAE
    Extrae_event (1000, 3);
    #endif
    PetscCall(readUserParams(&params, size));
    #ifdef USE_EXTRAE
    Extrae_event (1000, 0);
    #endif
    
    /* Create and setup source */
    #ifdef USE_EXTRAE
    Extrae_event (1000, 4);
    #endif
    PetscCall(setupSource(&sources, params));
    #ifdef USE_EXTRAE
    Extrae_event (1000, 0);
    #endif

    /* Import mesh and resistivity model */    
    #ifdef USE_EXTRAE
    Extrae_event (1000, 5);
    #endif    
    PetscCall(importGrid(&dm, &resistivity, params));
    #ifdef USE_EXTRAE
    Extrae_event (1000, 0);
    #endif

    /* Setup grid for FE computations */
    #ifdef USE_EXTRAE
    Extrae_event (1000, 6);
    #endif
    PetscCall(setupGrid(&dm, &grid, params));
    #ifdef USE_EXTRAE
    Extrae_event (1000, 0);
    #endif

    /* Setup linear system */
    #ifdef USE_EXTRAE
    Extrae_event (1000, 7);
    #endif
    /* Assemble linear system */
    PetscCall(assembleSystem(dm, resistivity, grid, sources, params, &A, &B, &G));
    #ifdef USE_EXTRAE
    Extrae_event (1000, 0);
    #endif

    // /* Solve linear system */
    #ifdef USE_EXTRAE
    Extrae_event (1000, 8);
    #endif
    PetscCall(solveSystem(dm, A, B, G, params, &X));
    #ifdef USE_EXTRAE
    Extrae_event (1000, 0);
    #endif

    // /* Postprocessing solution */
    #ifdef USE_EXTRAE
    Extrae_event (1000, 9);
    #endif
    PetscCall(computeFields(dm, X, grid, sources, params));
    #ifdef USE_EXTRAE
    Extrae_event (1000, 0);
    #endif

    // /* Print PETGEM footer */
    #ifdef USE_EXTRAE
    Extrae_event (1000, 10);
    #endif
    PetscCall(printFooter());
    #ifdef USE_EXTRAE
    Extrae_event (1000, 0);
    #endif

    /* Free memory */
    PetscCall(DMDestroy(&grid.H1dm));
    PetscCall(DMDestroy(&dm));
    PetscCall(VecDestroy(&resistivity));
    PetscCall(MatDestroy(&G));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&B));
    PetscCall(MatDestroy(&X));
    PetscCall(PetscFree(sources.sourceArray));
 
    /* PETSc finalize*/
    PetscCall(PetscFinalize());
    return 0;
}
