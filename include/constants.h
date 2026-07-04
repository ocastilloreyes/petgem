/*
 * Filename: constants.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-03-04
 *
 * Description:
 * Constants used throughout the PETGEM project.
 */

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <petsc.h>

/** @brief Magnetic permeability of free space μ₀ (H/m). */
#define MU (4.0 * PETSC_PI * 1.0e-7)

/** @brief Reference tetrahedral-cell topology counts and fixed array sizes. */
#define NUM_FACES_PER_CELL 4           /**< Faces per tetrahedron. */
#define NUM_EDGES_PER_CELL 6           /**< Edges per tetrahedron. */
#define NUM_VERTICES_PER_CELL 4        /**< Vertices per tetrahedron. */
#define NUM_EDGES_PER_FACE 3           /**< Edges per triangular face. */
#define NUM_VERTICES_PER_EDGE 2        /**< Vertices per edge. */
#define NUM_VERTICES_PER_FACE 3        /**< Vertices per triangular face. */
#define NUM_DIMENSIONS 3               /**< Spatial dimensions. */
#define NUM_H1_DOF_PER_CELL 4          /**< P1 H1 (vertex) DOFs per cell. */
#define NUM_CONDUCTIVITY_COMPONENTS 3  /**< Conductivity components (σx, σy, σz). */
#define NUM_MATERIALS_ID_COMPONENTS 1  /**< Material-id components per cell. */
#define MAX_TRANSITIVE_CLOSURE_SIZE 90 /**< Upper bound on DMPlex closure size. */
#define INV_VTU_NUM_FIELDS 1           /**< Cell-data fields written per VTU piece (rho_ohm_m). */
#define FE_NODAL_MAX_ORDER 64		   /**< Highest supported order; sizes the 1D factor tables. */

/* Largest DOF counts over the input order range (order = 1..6). These size only
   the small per-dof work arrays (O(dof) each, ~40 KB on the stack at order=6); the
   O(dof^2) gradient work arrays are heap-allocated to the actual dof. */
#define FEM_MAX_ND_DOF 216  /* order*(order+2)*(order+3)/2     at order=6 */
#define FEM_MAX_H1_DOF 84   /* (order+1)*(order+2)*(order+3)/6 at order=6 */

/**
 * @brief Inversion-kernel array caps (compile-time constants).
 *
 * Sized at struct-definition time so they must be compile-time constants.
 * Bump and recompile if a use case exceeds these; setupInversionSources
 * errors when exceeded, and the fixed-materials option array is sized to
 * the cap on read.
 */
#define INV_MAX_FIXED_MATERIALS  8  /**< Max number of held-fixed materials. */
#define INV_MAX_FREQUENCIES      64 /**< Max number of inversion frequencies. */

/** @brief Reference-cell local edge-to-vertex table (defined in constants.c). */
extern const PetscInt EDGE_VERTICES[NUM_EDGES_PER_CELL][NUM_VERTICES_PER_EDGE];

#endif
