/*
 * Filename: constants.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-03-04
 *
 * Description:
 * Constant definitions used throughout PETGEM.
 */

/* C libraries */

/* PETSc libraries */
#include <petscsys.h>

/* PETGEM functions */
#include "constants.h"

/**
 * @brief Edge-to-vertex mapping for a tetrahedral reference cell.
 * @details Defines which two vertices form each edge in a tetrahedron.
 * The edges are numbered from 0 to 5, corresponding to:
 *   - Edge 0: vertex 0 --> vertex 1
 *   - Edge 1: vertex 1 --> vertex 2
 *   - Edge 2: vertex 2 --> vertex 0
 *   - Edge 3: vertex 0 --> vertex 3
 *   - Edge 4: vertex 3 --> vertex 1
 *   - Edge 5: vertex 2 --> vertex 3
 * Used when reporting per-edge connectivity and midpoints.
 */
const PetscInt EDGE_VERTICES[NUM_EDGES_PER_CELL][NUM_VERTICES_PER_EDGE] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {3, 1}, {2, 3}};
