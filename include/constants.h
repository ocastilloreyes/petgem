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

/* Physical constants */
#define MU (4.0 * PETSC_PI * 1.0e-7)

/* Reference cell */
#define NUM_FACES_PER_CELL 4
#define NUM_EDGES_PER_CELL 6
#define NUM_VERTICES_PER_CELL 4
#define NUM_EDGES_PER_FACE 3
#define NUM_VERTICES_PER_EDGE 2
#define NUM_VERTICES_PER_FACE 3
#define NUM_DIMENSIONS 3
#define NUM_H1_DOF_PER_CELL 4
#define NUM_CONDUCTIVITY_COMPONENTS 3
#define NUM_MATERIALS_ID_COMPONENTS 1
#define MAX_TRANSITIVE_CLOSURE_SIZE 90

/* Inversion-kernel array caps - sized at struct-definition time so they
 * must be compile-time constants. Bump and recompile if a use case
 * exceeds these. setupInversionSources errors when exceeded; the
 * fixed-materials option array is sized to the cap on read. */
#define INV_MAX_FIXED_MATERIALS  8
#define INV_MAX_FREQUENCIES      64
#define INV_MAX_FD_CHECK_CELLS   1024
extern const PetscInt EDGE_VERTICES[NUM_EDGES_PER_CELL][NUM_VERTICES_PER_EDGE];
extern const PetscInt FACE_VERTICES[NUM_FACES_PER_CELL][NUM_VERTICES_PER_FACE];
extern const PetscInt REFERENCE_CELL[NUM_DIMENSIONS][NUM_VERTICES_PER_CELL];

/* 1D quadrature gauss points */
extern const PetscReal NORD1_1DGAUSSPOINTS[1][2];
extern const PetscReal NORD2_1DGAUSSPOINTS[2][2];
extern const PetscReal NORD3_1DGAUSSPOINTS[3][2];
extern const PetscReal NORD4_1DGAUSSPOINTS[4][2];
extern const PetscReal NORD5_1DGAUSSPOINTS[5][2];
extern const PetscReal NORD6_1DGAUSSPOINTS[6][2];
extern const PetscReal NORD7_1DGAUSSPOINTS[7][2];
extern const PetscReal NORD8_1DGAUSSPOINTS[8][2];
extern const PetscReal NORD11_1DGAUSSPOINTS[11][2];

/* 2D quadrature gauss points */
extern const PetscReal NORD1_2DGAUSSPOINTS[1][3];
extern const PetscReal NORD2_2DGAUSSPOINTS[3][3];
extern const PetscReal NORD3_2DGAUSSPOINTS[4][3];
extern const PetscReal NORD4_2DGAUSSPOINTS[6][3];
extern const PetscReal NORD5_2DGAUSSPOINTS[7][3];
extern const PetscReal NORD6_2DGAUSSPOINTS[12][3];
extern const PetscReal NORD7_2DGAUSSPOINTS[13][3];
extern const PetscReal NORD8_2DGAUSSPOINTS[15][3];
extern const PetscReal NORD9_2DGAUSSPOINTS[19][3];
extern const PetscReal NORD10_2DGAUSSPOINTS[25][3];
extern const PetscReal NORD11_2DGAUSSPOINTS[27][3];
extern const PetscReal NORD12_2DGAUSSPOINTS[33][3];
extern const PetscReal NORD13_2DGAUSSPOINTS[37][3];
extern const PetscReal NORD14_2DGAUSSPOINTS[42][3];
extern const PetscReal NORD15_2DGAUSSPOINTS[48][3];
extern const PetscReal NORD16_2DGAUSSPOINTS[52][3];
extern const PetscReal NORD17_2DGAUSSPOINTS[61][3];
extern const PetscReal NORD18_2DGAUSSPOINTS[70][3];
extern const PetscReal NORD19_2DGAUSSPOINTS[73][3];

/* 3D quadrature gauss points */
extern const PetscReal NORD1_3DGAUSSPOINTS[1][4];
extern const PetscReal NORD2_3DGAUSSPOINTS[4][4];
extern const PetscReal NORD3_3DGAUSSPOINTS[5][4];
extern const PetscReal NORD4_3DGAUSSPOINTS[11][4];
extern const PetscReal NORD5_3DGAUSSPOINTS[14][4];
extern const PetscReal NORD6_3DGAUSSPOINTS[24][4];
extern const PetscReal NORD7_3DGAUSSPOINTS[31][4];
extern const PetscReal NORD8_3DGAUSSPOINTS[43][4];
extern const PetscReal NORD9_3DGAUSSPOINTS[53][4];
extern const PetscReal NORD10_3DGAUSSPOINTS[126][4];
extern const PetscReal NORD12_3DGAUSSPOINTS[210][4];

#endif
