/*
 * Filename: hvfem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-06-12
 *
 * Description:
 * This file contains functions for high-order vector finite element method (HVFEM) computations. 
 *
 * Usage:
 * Include this file in your source code to utilize the grid functions. 
 * For example:
 * #include "grid.h"
 *
*/

/* C libraries */ 

/* PETSc libraries */
#include <petsc.h>
#include <petscsys.h> 

/* PETGEM functions */ 
#include "constants.h"


/**
 * @brief Transforms global XYZ coordinates to reference tetrahedron coordinates (Xi, Eta, Zeta).
 * @param[in] cellCoords Spatial coordinates of the tetrahedron's 4 vertices (PetscScalar array, size 12).
 * @param[in] point The global [x, y, z] coordinates of the point to transform.
 * @param[out] XiEtaZeta The resulting reference coordinates [xi, eta, zeta].
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Computes the inverse of the affine mapping from the reference tetrahedron
 *          (vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)) to the physical tetrahedron defined
 *          by `cellCoords`. Uses Cramer's rule / determinant formulas.
 */
PetscErrorCode tetrahedronXYZToXiEtaZeta(PetscScalar *cellCoords, PetscReal point[NUM_DIMENSIONS], PetscReal XiEtaZeta[NUM_DIMENSIONS]){
    
    PetscFunctionBeginUser;

    PetscReal J, xi, eta, zeta, cellCoordsReal[NUM_VERTICES_PER_ELEMENT*NUM_DIMENSIONS];

    for (PetscInt i=0; i < NUM_VERTICES_PER_ELEMENT*NUM_DIMENSIONS; i++){
        cellCoordsReal[i] = PetscRealPart(cellCoords[i]);
    }

    J = cellCoordsReal[5] * ( cellCoordsReal[0] * (cellCoordsReal[10] - cellCoordsReal[7])
        + cellCoordsReal[6] * (cellCoordsReal[1] - cellCoordsReal[10])
        + cellCoordsReal[9] * (cellCoordsReal[7] - cellCoordsReal[1]) )
        + cellCoordsReal[2] * ( cellCoordsReal[3] * (cellCoordsReal[7] - cellCoordsReal[10])
        + cellCoordsReal[6] * (cellCoordsReal[10] - cellCoordsReal[4])
        + cellCoordsReal[9] * (cellCoordsReal[4] - cellCoordsReal[7]) )
        + cellCoordsReal[8] * ( cellCoordsReal[3] * (cellCoordsReal[10] - cellCoordsReal[1])
        + cellCoordsReal[0] * (cellCoordsReal[4] - cellCoordsReal[10])
        + cellCoordsReal[9] * (cellCoordsReal[1] - cellCoordsReal[4]) )
        + cellCoordsReal[11] * ( cellCoordsReal[3] * (cellCoordsReal[1] - cellCoordsReal[7])
        + cellCoordsReal[0] * (cellCoordsReal[7] - cellCoordsReal[4])
        + cellCoordsReal[6] * (cellCoordsReal[4] - cellCoordsReal[1]) );

    /* Compute affine transformation for xi */
    xi = ( cellCoordsReal[11] * (cellCoordsReal[7] - cellCoordsReal[4]) + cellCoordsReal[5] * (cellCoordsReal[10] - cellCoordsReal[7])
         + cellCoordsReal[8] * (cellCoordsReal[4] - cellCoordsReal[10]) ) / J * point[0] +
         ( cellCoordsReal[5] * (cellCoordsReal[6] - cellCoordsReal[9]) + cellCoordsReal[11] * (cellCoordsReal[3] - cellCoordsReal[6])
         + cellCoordsReal[8] * (cellCoordsReal[9] - cellCoordsReal[3]) ) / J * point[1] +
         ( cellCoordsReal[3] * (cellCoordsReal[7] - cellCoordsReal[10]) + cellCoordsReal[9] * (cellCoordsReal[4] - cellCoordsReal[7])
         + cellCoordsReal[6] * (cellCoordsReal[10] - cellCoordsReal[4]) ) / J * point[2] +
         ( cellCoordsReal[8] * (cellCoordsReal[3] * cellCoordsReal[10] - cellCoordsReal[9] * cellCoordsReal[4])
         + cellCoordsReal[5] * (cellCoordsReal[9] * cellCoordsReal[7] - cellCoordsReal[6] * cellCoordsReal[10])
         + cellCoordsReal[11] * (cellCoordsReal[6] * cellCoordsReal[4] - cellCoordsReal[3] * cellCoordsReal[7]) ) / J;
        
    /* Compute affine transformation for eta */
    eta = ( cellCoordsReal[2] * (cellCoordsReal[10] - cellCoordsReal[4]) + cellCoordsReal[11] * (cellCoordsReal[4] - cellCoordsReal[1])
          + cellCoordsReal[5] * (cellCoordsReal[1] - cellCoordsReal[10]) ) / J * point[0] +
          ( cellCoordsReal[2] * (cellCoordsReal[3] - cellCoordsReal[9]) + cellCoordsReal[5] * (cellCoordsReal[9] - cellCoordsReal[0])
          + cellCoordsReal[11] * (cellCoordsReal[0] - cellCoordsReal[3]) ) / J * point[1] +
          ( cellCoordsReal[0] * (cellCoordsReal[4] - cellCoordsReal[10]) + cellCoordsReal[3] * (cellCoordsReal[10] - cellCoordsReal[1])
          + cellCoordsReal[9] * (cellCoordsReal[1] - cellCoordsReal[4]) ) / J * point[2] +
          ( cellCoordsReal[2] * (cellCoordsReal[9] * cellCoordsReal[4] - cellCoordsReal[3] * cellCoordsReal[10])
          + cellCoordsReal[5] * (cellCoordsReal[0] * cellCoordsReal[10] - cellCoordsReal[9] * cellCoordsReal[1])
          + cellCoordsReal[11] * (cellCoordsReal[3] * cellCoordsReal[1] - cellCoordsReal[0] * cellCoordsReal[4]) ) / J;
        
    /* Compute affine transformation for zeta */
    zeta = ( cellCoordsReal[5] * (cellCoordsReal[7] - cellCoordsReal[1]) + cellCoordsReal[8] * (cellCoordsReal[1] - cellCoordsReal[4])
           + cellCoordsReal[2] * (cellCoordsReal[4] - cellCoordsReal[7]) ) / J * point[0] +
           ( cellCoordsReal[8] * (cellCoordsReal[3] - cellCoordsReal[0]) + cellCoordsReal[5] * (cellCoordsReal[0] - cellCoordsReal[6])
           + cellCoordsReal[2] * (cellCoordsReal[6] - cellCoordsReal[3]) ) / J * point[1] +
           ( cellCoordsReal[3] * (cellCoordsReal[1] - cellCoordsReal[7]) + cellCoordsReal[0] * (cellCoordsReal[7] - cellCoordsReal[4])
           + cellCoordsReal[6] * (cellCoordsReal[4] - cellCoordsReal[1]) ) / J * point[2] +
           ( cellCoordsReal[5] * ( cellCoordsReal[9] * (cellCoordsReal[1] - cellCoordsReal[7])
           + cellCoordsReal[10] * (cellCoordsReal[6] - cellCoordsReal[0]) )
           + cellCoordsReal[8] * ( cellCoordsReal[9] * (cellCoordsReal[4] - cellCoordsReal[1])
           + cellCoordsReal[10] * (cellCoordsReal[0] - cellCoordsReal[3]) )
           + cellCoordsReal[2] * ( cellCoordsReal[9] * (cellCoordsReal[7] - cellCoordsReal[4])
           + cellCoordsReal[10] * (cellCoordsReal[3] - cellCoordsReal[6]) )
           + cellCoordsReal[11] * ( cellCoordsReal[0] * (cellCoordsReal[4] - cellCoordsReal[7])
           + cellCoordsReal[3] * (cellCoordsReal[7] - cellCoordsReal[1])
           + cellCoordsReal[6] * (cellCoordsReal[1] - cellCoordsReal[4]) ) + J ) / J;
            
    XiEtaZeta[0] = xi;
    XiEtaZeta[1] = eta;
    XiEtaZeta[2] = zeta;
    
    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes a rotation vector based on azimuth and dip angles.
 * @param[in] azimuth Rotation angle in the x-y plane (degrees).
 * @param[in] dip Rotation angle in the x-z plane (degrees).
 * @param[out] rotationVector The resulting 3D unit vector after rotation.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Starts with a base vector [1, 0, 0]. Converts azimuth and dip to radians.
 *          Applies rotation matrices sequentially (x-y plane first, then x-z plane).
 *          A y-z plane rotation (tetha) is included but currently hardcoded to 0 degrees.
 *          The final rotated vector is stored in `rotationVector`.
 */
PetscErrorCode vectorRotation(PetscReal azimuth, PetscReal dip, PetscReal rotationVector[NUM_DIMENSIONS]){

    PetscFunctionBeginUser;

    PetscReal base_vector[NUM_DIMENSIONS] = {1., 0., 0.};

    /* Convert degrees to radians for rotation */
    PetscReal alpha = azimuth * PETSC_PI / 180.;    // x-y plane
    PetscReal beta  = dip * PETSC_PI / 180.;        // x-z plane
    PetscReal tetha = 0.0 * PETSC_PI / 180.;        // y-z plane

    /* Define rotation matrices for each plane */
    /* x-y plane */
    PetscReal M1[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{PetscCosReal(alpha), -PetscSinReal(alpha),   0.},
                                                    {PetscSinReal(alpha),  PetscCosReal(alpha),   0.},
                                                    {                 0.,                   0.,   1.}};

    /* x-z plane */
    PetscReal M2[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{PetscCosReal(beta),  0.,  -PetscSinReal(beta)},
                                                    {                0.,  1.,                   0.},
                                                    {PetscSinReal(beta),  0.,   PetscCosReal(beta)}};

    /* y-z plane */
    PetscReal M3[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{1.,   0.,                                     0.},
                                                    {0.,   PetscCosReal(tetha),  -PetscSinReal(tetha)},
                                                    {0.,   PetscSinReal(tetha),   PetscCosReal(tetha)}};
    
    PetscReal temp1[NUM_DIMENSIONS][NUM_DIMENSIONS], temp2[NUM_DIMENSIONS][NUM_DIMENSIONS];

    /* Perform matrix multiplications */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            PetscReal sum = 0.0;
            for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
                sum += M1[i][k] * M2[k][j];
            }
            temp1[i][j] = sum;
        }
    }

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            PetscReal sum = 0.0;
            for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
                sum += temp1[i][k] * M3[k][j];
            }
            temp2[i][j] = sum;
        }
    }

    /* Apply rotation */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        rotationVector[i] = 0.0;
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            rotationVector[i] += temp2[i][j] * base_vector[j];
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes the cross product of two 3D vectors.
 * @param[in] vector1 The first input vector [x1, y1, z1].
 * @param[in] vector2 The second input vector [x2, y2, z2].
 * @param[out] result The resulting cross product vector [x, y, z].
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Calculates result = vector1 x vector2 using the standard formula.
 */
PetscErrorCode crossProduct(PetscReal vector1[NUM_DIMENSIONS], PetscReal vector2[NUM_DIMENSIONS], PetscReal result[NUM_DIMENSIONS]){
    PetscFunctionBeginUser;

    /* Compute x component */
    result[0] = vector1[1] * vector2[2] - vector1[2] * vector2[1]; 

    /* Compute y component */
    result[1] = vector1[2] * vector2[0] - vector1[0] * vector2[2];

    /* Compute z component */
    result[2] = vector1[0] * vector2[1] - vector1[1] * vector2[0];

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the product of a 3x3 matrix and a 3D column vector (result = matrix * vector).
 * @param[in] vector The input column vector [v1, v2, v3].
 * @param[in] matrix The input 3x3 matrix.
 * @param[out] result The resulting 3D column vector.
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode matrixVectorProduct(PetscReal vector[NUM_DIMENSIONS], PetscReal matrix[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal result[NUM_DIMENSIONS]){
    PetscFunctionBeginUser;

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        result[i] = 0.0;
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++){
            result[i] += matrix[i][j] * vector[j];
        }
    }    

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes the product of a 3D row vector and a 3x3 matrix (result = vector * matrix).
 * @param[in] vector The input row vector [v1, v2, v3].
 * @param[in] matrix The input 3x3 matrix.
 * @param[out] result The resulting 3D row vector.
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode vectorMatrixProduct(PetscReal vector[NUM_DIMENSIONS], PetscReal matrix[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal result[NUM_DIMENSIONS]){
    PetscFunctionBeginUser;

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        result[i] = 0.0;
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++){
            result[i] += matrix[j][i] * vector[j];
        }
    }    

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes the dot product of two 3D vectors.
 * @param[in] vector1 The first input vector [x1, y1, z1].
 * @param[in] vector2 The second input vector [x2, y2, z2].
 * @param[out] result Pointer to the scalar result.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Calculates *result = vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2].
 */
PetscErrorCode dotProduct(PetscReal vector1[NUM_DIMENSIONS], PetscReal vector2[NUM_DIMENSIONS], PetscReal *result){
    PetscFunctionBeginUser;

    *result = 0.0;
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        *result += vector1[i] * vector2[i];
    }    

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes the Jacobian matrix and its inverse for the affine mapping from the reference tetrahedron to the physical tetrahedron.
 * @param[in] cellCoords Spatial coordinates of the tetrahedron's 4 vertices (PetscScalar array, size 12).
 * @param[out] jacobian The 3x3 Jacobian matrix.
 * @param[out] invJacobian The 3x3 inverse Jacobian matrix.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Calculates the Jacobian matrix based on the differences between vertex coordinates.
 *          Computes the determinant, cofactor matrix, adjugate matrix, and finally the inverse Jacobian.
 */
PetscErrorCode computeJacobian(PetscScalar *cellCoords, PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS]) {
    PetscFunctionBeginUser;

    PetscReal determinant;     
    PetscReal coFactorMatrix[NUM_DIMENSIONS][NUM_DIMENSIONS], adjugateMatrix[NUM_DIMENSIONS][NUM_DIMENSIONS];
    PetscReal invDeterminant;

    /* Reset matrices */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            jacobian[i][j] = 0.0;
            invJacobian[i][j] = 0.0;
        }
    }

    /* Compute jacobian */ 
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        jacobian[0][i] = PetscRealPart(cellCoords[i])   - PetscRealPart(cellCoords[3+i]); 
        jacobian[1][i] = PetscRealPart(cellCoords[6+i]) - PetscRealPart(cellCoords[3+i]); 
        jacobian[2][i] = PetscRealPart(cellCoords[9+i]) - PetscRealPart(cellCoords[3+i]);
    }

    /* Compute determinant */
    determinant = jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1]) -
                  jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0]) +
                  jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]);

    /* Compute cofactor matrix */
    coFactorMatrix[0][0] =   jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1];
    coFactorMatrix[0][1] = -(jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0]);
    coFactorMatrix[0][2] =   jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0];
    coFactorMatrix[1][0] = -(jacobian[0][1] * jacobian[2][2] - jacobian[0][2] * jacobian[2][1]);
    coFactorMatrix[1][1] =   jacobian[0][0] * jacobian[2][2] - jacobian[0][2] * jacobian[2][0];
    coFactorMatrix[1][2] = -(jacobian[0][0] * jacobian[2][1] - jacobian[0][1] * jacobian[2][0]);
    coFactorMatrix[2][0] =   jacobian[0][1] * jacobian[1][2] - jacobian[0][2] * jacobian[1][1];
    coFactorMatrix[2][1] = -(jacobian[0][0] * jacobian[1][2] - jacobian[0][2] * jacobian[1][0]);
    coFactorMatrix[2][2] =   jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];

    /* Compute adjugate matrix */ 
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            adjugateMatrix[i][j] = coFactorMatrix[j][i];
        }
    }

    /* Compute inverse of jacobian */
    invDeterminant = 1.0 / determinant;            
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            invJacobian[i][j] = invDeterminant * adjugateMatrix[i][j];
        }
    }
    
    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Determines the number of Gauss points and its Weights required for integrating polynomials
 *        up to a given order on a tetrahedron.
 *
 * @param[in] nord The basis order (determines the required integration order 2*params.nord).
 * @param[out] numGaussPoints Pointer to store the required number of Gauss points.
 * @return PetscErrorCode PETSC_SUCCESS on success. Returns error code if the required Gauss order (2*nord)
 *         is out of the supported range [1, 12].
 */
PetscErrorCode computeNumGaussPoints3D(PetscInt nord, PetscInt *numGaussPoints){
    PetscFunctionBeginUser;

    PetscInt gaussOrder;
                
    /* Compute gauss order */ 
    gaussOrder = 2*nord;
            
    PetscCheck(gaussOrder >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: Orders lower than 1 are not supported.\n");
    PetscCheck(gaussOrder <= 12, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: Orders higher than 6 are not supported by PETGEM.\n");

    switch (gaussOrder) {
        case 1:  *numGaussPoints = 1; break;
        case 2:  *numGaussPoints = 4; break;
        case 3:  *numGaussPoints = 5; break;
        case 4:  *numGaussPoints = 11; break;
        case 5:  *numGaussPoints = 14; break;
        case 6:  *numGaussPoints = 24; break;
        case 7:  *numGaussPoints = 31; break;
        case 8:  *numGaussPoints = 43; break;
        case 9:  *numGaussPoints = 53; break;
        case 10: *numGaussPoints = 126; break;
        case 11: *numGaussPoints = 126; break;
        case 12: *numGaussPoints = 210; break;
        default: break;    
    }
    
    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Renormalizes Gauss points from a [-1, 1]-based domain to the [0, 1]-based reference tetrahedron.
 *
 * @param[in] numPoints The number of Gauss points.
 * @param[in] gaussPoints Input array of points.
 * @param[out] points Output array (numPoints x NUM_DIMENSIONS) for the renormalized coordinates [xi, eta, zeta].
 * @param[out] weights Output array (numPoints) for the renormalized weights.
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode renormalization3DGaussPoints(PetscInt numPoints, const PetscReal (*gaussPoints)[4], PetscReal** points, PetscReal* weights){
    PetscFunctionBeginUser;

    for(PetscInt i = 0; i < numPoints; i++){
        weights[i] = gaussPoints[i][NUM_DIMENSIONS]/8;
        points[i][0] = (1 + gaussPoints[i][1])/2;
        points[i][1] = -(1 + gaussPoints[i][0] + gaussPoints[i][1] + gaussPoints[i][2])/2;
        points[i][2] = (1 + gaussPoints[i][0])/2;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Provides coordinates and weights for Gauss quadrature on the reference tetrahedron for various orders.
 *
 * @param numPoints [in] The desired number of Gauss points.
 * @param points [out] Output array (numPoints x NUM_DIMENSIONS) for the coordinates [xi, eta, zeta].
 * @param weights [out] Output array (numPoints) for the weights.
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode computeGaussPoints3D(PetscInt numPoints, PetscReal **points, PetscReal *weights){
    PetscFunctionBeginUser;

    const PetscReal nord1_3DGaussPoints[1][4] = {{-0.500000000000000, -0.500000000000000, -0.500000000000000, 1.333333333333333}};

    const PetscReal nord2_3DGaussPoints[4][4] = {{-0.723606797749979, -0.723606797749979, -0.723606797749979, 0.333333333333333},
                                                 { 0.170820393249937, -0.723606797749979, -0.723606797749979, 0.333333333333333},
                                                 {-0.723606797749979,  0.170820393249937, -0.723606797749979, 0.333333333333333},
                                                 {-0.723606797749979, -0.723606797749979,  0.170820393249937, 0.333333333333333}};

    const PetscReal nord3_3DGaussPoints[5][4] = {{-0.500000000000000, -0.500000000000000, -0.500000000000000, -1.066666666666667},
                                                 {-0.666666666666667, -0.666666666666667, -0.666666666666667,  0.600000000000000},
                                                 {-0.666666666666667, -0.666666666666667,  0.000000000000000,  0.600000000000000},
                                                 {-0.666666666666667,  0.000000000000000, -0.666666666666667,  0.600000000000000},
                                                 { 0.000000000000000, -0.666666666666667, -0.666666666666667,  0.600000000000000}};

    const PetscReal nord4_3DGaussPoints[11][4] = {{-0.500000000000000, -0.500000000000000, -0.500000000000000, -0.105244444444444},
                                                  {-0.857142857142857, -0.857142857142857, -0.857142857142857,  0.060977777777778},
                                                  {-0.857142857142857, -0.857142857142857,  0.571428571428571,  0.060977777777778},
                                                  {-0.857142857142857,  0.571428571428571, -0.857142857142857,  0.060977777777778},
                                                  { 0.571428571428571, -0.857142857142857, -0.857142857142857,  0.060977777777778},
                                                  {-0.201192847666402, -0.201192847666402, -0.798807152333598,  0.199111111111111},
                                                  {-0.201192847666402, -0.798807152333598, -0.201192847666402,  0.199111111111111},
                                                  {-0.798807152333598, -0.201192847666402, -0.201192847666402,  0.199111111111111},
                                                  {-0.201192847666402, -0.798807152333598, -0.798807152333598,  0.199111111111111},
                                                  {-0.798807152333598, -0.201192847666402, -0.798807152333598,  0.199111111111111},
                                                  {-0.798807152333598, -0.798807152333598, -0.201192847666402,  0.199111111111111}};

    const PetscReal nord5_3DGaussPoints[14][4] = {{-0.814529499378218, -0.814529499378218, -0.814529499378218,  0.097990724155149},
                                                  { 0.443588498134653, -0.814529499378218, -0.814529499378218,  0.097990724155149},
                                                  {-0.814529499378218,  0.443588498134653, -0.814529499378218,  0.097990724155149},
                                                  {-0.814529499378218, -0.814529499378218,  0.443588498134653,  0.097990724155149},
                                                  {-0.378228161473399, -0.378228161473399, -0.378228161473399,  0.150250567624021},
                                                  {-0.865315515579804, -0.378228161473399, -0.378228161473399,  0.150250567624021},
                                                  {-0.378228161473399, -0.865315515579804, -0.378228161473399,  0.150250567624021},
                                                  {-0.378228161473399, -0.378228161473399, -0.865315515579804,  0.150250567624021},
                                                  {-0.091007408251299, -0.091007408251299, -0.908992591748701,  0.056728027702775},
                                                  {-0.091007408251299, -0.908992591748701, -0.091007408251299,  0.056728027702775},
                                                  {-0.908992591748701, -0.091007408251299, -0.091007408251299,  0.056728027702775},
                                                  {-0.091007408251299, -0.908992591748701, -0.908992591748701,  0.056728027702775},
                                                  {-0.908992591748701, -0.091007408251299, -0.908992591748701,  0.056728027702775},
                                                  {-0.908992591748701, -0.908992591748701, -0.091007408251299,  0.056728027702775}};

    const PetscReal nord6_3DGaussPoints[24][4] = {{-0.570794257481696, -0.570794257481696, -0.570794257481696, 0.053230333677557},
                                                  {-0.287617227554912, -0.570794257481696, -0.570794257481696, 0.053230333677557},
                                                  {-0.570794257481696, -0.287617227554912, -0.570794257481696, 0.053230333677557},
                                                  {-0.570794257481696, -0.570794257481696, -0.287617227554912, 0.053230333677557},
                                                  {-0.918652082930777, -0.918652082930777, -0.918652082930777, 0.013436281407094},
                                                  { 0.755956248792332, -0.918652082930777, -0.918652082930777, 0.013436281407094},
                                                  {-0.918652082930777,  0.755956248792332, -0.918652082930777 ,0.013436281407094},
                                                  {-0.918652082930777, -0.918652082930777,  0.755956248792332 ,0.013436281407094},
                                                  {-0.355324219715449, -0.355324219715449, -0.355324219715449, 0.073809575391540},
                                                  {-0.934027340853653, -0.355324219715449, -0.355324219715449, 0.073809575391540},
                                                  {-0.355324219715449, -0.934027340853653, -0.355324219715449, 0.073809575391540},
                                                  {-0.355324219715449, -0.355324219715449, -0.934027340853653, 0.073809575391540},
                                                  {-0.872677996249965, -0.872677996249965, -0.460655337083368, 0.064285714285714},
                                                  {-0.872677996249965, -0.460655337083368, -0.872677996249965, 0.064285714285714},
                                                  {-0.872677996249965, -0.872677996249965,  0.206011329583298, 0.064285714285714},
                                                  {-0.872677996249965,  0.206011329583298, -0.872677996249965, 0.064285714285714},
                                                  {-0.872677996249965, -0.460655337083368,  0.206011329583298, 0.064285714285714},
                                                  {-0.872677996249965,  0.206011329583298, -0.460655337083368, 0.064285714285714},
                                                  {-0.460655337083368, -0.872677996249965, -0.872677996249965, 0.064285714285714},
                                                  {-0.460655337083368, -0.872677996249965,  0.206011329583298, 0.064285714285714},
                                                  {-0.460655337083368,  0.206011329583298, -0.872677996249965, 0.064285714285714},
                                                  { 0.206011329583298, -0.872677996249965, -0.460655337083368, 0.064285714285714},
                                                  { 0.206011329583298, -0.872677996249965, -0.872677996249965, 0.064285714285714},
                                                  { 0.206011329583298, -0.460655337083368, -0.872677996249965, 0.064285714285714}};

    const PetscReal nord7_3DGaussPoints[31][4] = {{ 0.000000000000000,   0.000000000000000,  -1.000000000000000,   0.007760141093474},
                                                  { 0.000000000000000,  -1.000000000000000,   0.000000000000000,   0.007760141093474},
                                                  {-1.000000000000000,   0.000000000000000,   0.000000000000000,   0.007760141093474},
                                                  {-1.000000000000000,  -1.000000000000000,   0.000000000000000,   0.007760141093474},
                                                  {-1.000000000000000,   0.000000000000000,  -1.000000000000000,   0.007760141093474},
                                                  { 0.000000000000000,  -1.000000000000000,  -1.000000000000000,   0.007760141093474},
                                                  {-0.500000000000000,  -0.500000000000000,  -0.500000000000000,   0.146113787728871},
                                                  {-0.843573615339364,  -0.843573615339364,  -0.843573615339364,   0.084799532195309},
                                                  {-0.843573615339364,  -0.843573615339364,   0.530720846018092,   0.084799532195309},
                                                  {-0.843573615339364,   0.530720846018092,  -0.843573615339364,   0.084799532195309},
                                                  { 0.530720846018092,  -0.843573615339364,  -0.843573615339364,   0.084799532195309},
                                                  {-0.756313566672190,  -0.756313566672190,  -0.756313566672190,  -0.500141920914655},
                                                  {-0.756313566672190,  -0.756313566672190,   0.268940700016569,  -0.500141920914655},
                                                  {-0.756313566672190,   0.268940700016569,  -0.756313566672190,  -0.500141920914655},
                                                  { 0.268940700016569,  -0.756313566672190,  -0.756313566672190,  -0.500141920914655},
                                                  {-0.334921671107159,  -0.334921671107159,  -0.334921671107159,   0.039131402104588},
                                                  {-0.334921671107159,  -0.334921671107159,  -0.995234986678524,   0.039131402104588},
                                                  {-0.334921671107159,  -0.995234986678524,  -0.334921671107159,   0.039131402104588},
                                                  {-0.995234986678524,  -0.334921671107159,  -0.334921671107159,   0.039131402104588},
                                                  {-0.800000000000000,  -0.800000000000000,  -0.600000000000000,   0.220458553791887},
                                                  {-0.800000000000000,  -0.600000000000000,  -0.800000000000000,   0.220458553791887},
                                                  {-0.800000000000000,  -0.800000000000000,   0.200000000000000,   0.220458553791887},
                                                  {-0.800000000000000,   0.200000000000000,  -0.800000000000000,   0.220458553791887},
                                                  {-0.800000000000000,  -0.600000000000000,   0.200000000000000,   0.220458553791887},
                                                  {-0.800000000000000,   0.200000000000000,  -0.600000000000000,   0.220458553791887},
                                                  {-0.600000000000000,  -0.800000000000000,  -0.800000000000000,   0.220458553791887},
                                                  {-0.600000000000000,  -0.800000000000000,   0.200000000000000,   0.220458553791887},
                                                  {-0.600000000000000,   0.200000000000000,  -0.800000000000000,   0.220458553791887},
                                                  { 0.200000000000000,  -0.800000000000000,  -0.600000000000000,   0.220458553791887},
                                                  { 0.200000000000000,  -0.800000000000000,  -0.800000000000000,   0.220458553791887},
                                                  { 0.200000000000000,  -0.600000000000000,  -0.800000000000000,   0.220458553791887}};

    const PetscReal nord8_3DGaussPoints[43][4] = {{-0.500000000000000,  -0.500000000000000,  -0.500000000000000, -0.164001509269119},
                                                  {-0.586340136778654,  -0.586340136778654,  -0.586340136778654,  0.114002446582935},
                                                  {-0.586340136778654,  -0.586340136778654,  -0.240979589664039,  0.114002446582935},
                                                  {-0.586340136778654,  -0.240979589664039,  -0.586340136778654,  0.114002446582935},
                                                  {-0.240979589664039,  -0.586340136778654,  -0.586340136778654,  0.114002446582935},
                                                  {-0.835792823378907,  -0.835792823378907,  -0.835792823378907,  0.015736266505071},
                                                  {-0.835792823378907,  -0.835792823378907,   0.507378470136720,  0.015736266505071},
                                                  {-0.835792823378907,   0.507378470136720,  -0.835792823378907,  0.015736266505071},
                                                  { 0.507378470136720,  -0.835792823378907,  -0.835792823378907,  0.015736266505071},
                                                  {-0.988436098989604,  -0.988436098989604,  -0.988436098989604,  0.001358672872743},
                                                  {-0.988436098989604,  -0.988436098989604,   0.965308296968812,  0.001358672872743},
                                                  {-0.988436098989604,   0.965308296968812,  -0.988436098989604,  0.001358672872743},
                                                  { 0.965308296968812,  -0.988436098989604,  -0.988436098989604,  0.001358672872743},
                                                  {-0.898934519962212,  -0.898934519962212,  -0.101065480037788,  0.036637470595738},
                                                  {-0.898934519962212,  -0.101065480037788,  -0.898934519962212,  0.036637470595738},
                                                  {-0.101065480037788,  -0.898934519962212,  -0.898934519962212,  0.036637470595738},
                                                  {-0.898934519962212,  -0.101065480037788,  -0.101065480037788,  0.036637470595738},
                                                  {-0.101065480037788,  -0.898934519962212,  -0.101065480037788,  0.036637470595738},
                                                  {-0.101065480037788,  -0.101065480037788,  -0.898934519962212,  0.036637470595738},
                                                  {-0.541866927766378,  -0.541866927766378,  -0.928720834422932,  0.045635886469455},
                                                  {-0.541866927766378,  -0.928720834422932,  -0.541866927766378,  0.045635886469455},
                                                  {-0.541866927766378,  -0.541866927766378,   0.012454689955687,  0.045635886469455},
                                                  {-0.541866927766378,   0.012454689955687,  -0.541866927766378,  0.045635886469455},
                                                  {-0.541866927766378,  -0.928720834422932,   0.012454689955687,  0.045635886469455},
                                                  {-0.541866927766378,   0.012454689955687,  -0.928720834422932,  0.045635886469455},
                                                  {-0.928720834422932,  -0.541866927766378,  -0.541866927766378,  0.045635886469455},
                                                  {-0.928720834422932,  -0.541866927766378,   0.012454689955687,  0.045635886469455},
                                                  {-0.928720834422932,   0.012454689955687,  -0.541866927766378,  0.045635886469455},
                                                  { 0.012454689955687,  -0.541866927766378,  -0.928720834422932,  0.045635886469455},
                                                  { 0.012454689955687,  -0.541866927766378,  -0.541866927766378,  0.045635886469455},
                                                  { 0.012454689955687,  -0.928720834422932,  -0.541866927766378,  0.045635886469455},
                                                  {-0.926784500893605,  -0.926784500893605,  -0.619027916130733,  0.017124153129297},
                                                  {-0.926784500893605,  -0.619027916130733,  -0.926784500893605,  0.017124153129297},
                                                  {-0.926784500893605,  -0.926784500893605,   0.472596917917943,  0.017124153129297},
                                                  {-0.926784500893605,   0.472596917917943,  -0.926784500893605,  0.017124153129297},
                                                  {-0.926784500893605,  -0.619027916130733,   0.472596917917943,  0.017124153129297},
                                                  {-0.926784500893605,   0.472596917917943,  -0.619027916130733,  0.017124153129297},
                                                  {-0.619027916130733,  -0.926784500893605,  -0.926784500893605,  0.017124153129297},
                                                  {-0.619027916130733,  -0.926784500893605,   0.472596917917943,  0.017124153129297},
                                                  {-0.619027916130733,   0.472596917917943,  -0.926784500893605,  0.017124153129297},
                                                  { 0.472596917917943,  -0.926784500893605,  -0.619027916130733,  0.017124153129297},
                                                  { 0.472596917917943,  -0.926784500893605,  -0.926784500893605,  0.017124153129297},
                                                  { 0.472596917917943,  -0.619027916130733,  -0.926784500893605,  0.017124153129297}};

    const PetscReal nord9_3DGaussPoints[53][4] = {{-0.500000000000000,  -0.500000000000000,  -0.500000000000000,  -1.102392306608869},
                                                  {-0.903297922900526,  -0.903297922900526,  -0.903297922900526,   0.014922692552682},
                                                  {-0.903297922900526,  -0.903297922900526,   0.709893768701580,   0.014922692552682},
                                                  {-0.903297922900526,   0.709893768701580,  -0.903297922900526,   0.014922692552682},
                                                  { 0.709893768701580,  -0.903297922900526,  -0.903297922900526,   0.014922692552682},
                                                  {-0.350841439764235,  -0.350841439764235,  -0.350841439764235,   0.034475391755947},
                                                  {-0.350841439764235,  -0.350841439764235,  -0.947475680707294,   0.034475391755947},
                                                  {-0.350841439764235,  -0.947475680707294,  -0.350841439764235,   0.034475391755947},
                                                  {-0.947475680707294,  -0.350841439764235,  -0.350841439764235,   0.034475391755947},
                                                  {-0.770766919552010,  -0.770766919552010,  -0.770766919552010,  -0.721478131849612},
                                                  {-0.770766919552010,  -0.770766919552010,   0.312300758656029,  -0.721478131849612},
                                                  {-0.770766919552010,   0.312300758656029,  -0.770766919552010,  -0.721478131849612},
                                                  { 0.312300758656029,  -0.770766919552010,  -0.770766919552010,  -0.721478131849612},
                                                  {-0.549020096176972,  -0.549020096176972,  -0.549020096176972,   0.357380609620092},
                                                  {-0.549020096176972,  -0.549020096176972,  -0.352939711469084,   0.357380609620092},
                                                  {-0.549020096176972,  -0.352939711469084,  -0.549020096176972,   0.357380609620092},
                                                  {-0.352939711469084,  -0.549020096176972,  -0.549020096176972,   0.357380609620092},
                                                  {-0.736744381506260,  -0.736744381506260,  -0.832670596765630,   0.277603247076406},
                                                  {-0.736744381506260,  -0.832670596765630,  -0.736744381506260,   0.277603247076406},
                                                  {-0.736744381506260,  -0.736744381506260,   0.306159359778151,   0.277603247076406},
                                                  {-0.736744381506260,   0.306159359778151,  -0.736744381506260,   0.277603247076406},
                                                  {-0.736744381506260,  -0.832670596765630,   0.306159359778151,   0.277603247076406},
                                                  {-0.736744381506260,   0.306159359778151,  -0.832670596765630,   0.277603247076406},
                                                  {-0.832670596765630,  -0.736744381506260,  -0.736744381506260,   0.277603247076406},
                                                  {-0.832670596765630,  -0.736744381506260,   0.306159359778151,   0.277603247076406},
                                                  {-0.832670596765630,   0.306159359778151,  -0.736744381506260,   0.277603247076406},
                                                  { 0.306159359778151,  -0.736744381506260,  -0.832670596765630,   0.277603247076406},
                                                  { 0.306159359778151,  -0.736744381506260,  -0.736744381506260,   0.277603247076406},
                                                  { 0.306159359778151,  -0.832670596765630,  -0.736744381506260,   0.277603247076406},
                                                  {-0.132097077177186,  -0.132097077177186,  -0.784460280901143,   0.026820671221285},
                                                  {-0.132097077177186,  -0.784460280901143,  -0.132097077177186,   0.026820671221285},
                                                  {-0.132097077177186,  -0.132097077177186,  -0.951345564744484,   0.026820671221285},
                                                  {-0.132097077177186,  -0.951345564744484,  -0.132097077177186,   0.026820671221285},
                                                  {-0.132097077177186,  -0.784460280901143,  -0.951345564744484,   0.026820671221285},
                                                  {-0.132097077177186,  -0.951345564744484,  -0.784460280901143,   0.026820671221285},
                                                  {-0.784460280901143,  -0.132097077177186,  -0.132097077177186,   0.026820671221285},
                                                  {-0.784460280901143,  -0.132097077177186,  -0.951345564744484,   0.026820671221285},
                                                  {-0.784460280901143,  -0.951345564744484,  -0.132097077177186,   0.026820671221285},
                                                  {-0.951345564744484,  -0.132097077177186,  -0.784460280901143,   0.026820671221285},
                                                  {-0.951345564744484,  -0.132097077177186,  -0.132097077177186,   0.026820671221285},
                                                  {-0.951345564744484,  -0.784460280901143,  -0.132097077177186,   0.026820671221285},
                                                  {-1.002752554636276,  -1.002752554636276,  -0.446893054726385,   0.003453031004456},
                                                  {-1.002752554636276,  -0.446893054726385,  -1.002752554636276,   0.003453031004456},
                                                  {-1.002752554636276,  -1.002752554636276,   0.452398163998938,   0.003453031004456},
                                                  {-1.002752554636276,   0.452398163998938,  -1.002752554636276,   0.003453031004456},
                                                  {-1.002752554636276,  -0.446893054726385,   0.452398163998938,   0.003453031004456},
                                                  {-1.002752554636276,   0.452398163998938,  -0.446893054726385,   0.003453031004456},
                                                  {-0.446893054726385,  -1.002752554636276,  -1.002752554636276,   0.003453031004456},
                                                  {-0.446893054726385,  -1.002752554636276,   0.452398163998938,   0.003453031004456},
                                                  {-0.446893054726385,   0.452398163998938,  -1.002752554636276,   0.003453031004456},
                                                  { 0.452398163998938,  -1.002752554636276,  -0.446893054726385,   0.003453031004456},
                                                  { 0.452398163998938,  -1.002752554636276,  -1.002752554636276,   0.003453031004456},
                                                  { 0.452398163998938,  -0.446893054726385,  -1.002752554636276,   0.003453031004456}};

    const PetscReal nord10_3DGaussPoints[126][4] = {{-0.857142857142857,  -0.857142857142857,  0.571428571428571 ,   0.362902592520648},
                                                    {-0.857142857142857,  -0.571428571428571,   0.285714285714286,   0.362902592520648},
                                                    {-0.857142857142857,  -0.285714285714286,   0.000000000000000,   0.362902592520648},
                                                    {-0.857142857142857,   0.000000000000000,  -0.285714285714286,   0.362902592520648},
                                                    {-0.857142857142857,   0.285714285714286,  -0.571428571428571,   0.362902592520648},
                                                    {-0.857142857142857,   0.571428571428571,  -0.857142857142857,   0.362902592520648},
                                                    {-0.571428571428571,  -0.857142857142857,   0.285714285714286,   0.362902592520648},
                                                    {-0.571428571428571,  -0.571428571428571,   0.000000000000000,   0.362902592520648},
                                                    {-0.571428571428571,  -0.285714285714286,  -0.285714285714286,   0.362902592520648},
                                                    {-0.571428571428571,   0.000000000000000,  -0.571428571428571,   0.362902592520648},
                                                    {-0.571428571428571,   0.285714285714286,  -0.857142857142857,   0.362902592520648},
                                                    {-0.285714285714286,  -0.857142857142857,   0.000000000000000,   0.362902592520648},
                                                    {-0.285714285714286,  -0.571428571428571,  -0.285714285714286,   0.362902592520648},
                                                    {-0.285714285714286,  -0.285714285714286,  -0.571428571428571,   0.362902592520648},
                                                    {-0.285714285714286,   0.000000000000000,  -0.857142857142857,   0.362902592520648},
                                                    { 0.000000000000000,  -0.857142857142857,  -0.285714285714286,   0.362902592520648},
                                                    { 0.000000000000000,  -0.571428571428571,  -0.571428571428571,   0.362902592520648},
                                                    { 0.000000000000000,  -0.285714285714286,  -0.857142857142857,   0.362902592520648},
                                                    { 0.285714285714286,  -0.857142857142857,  -0.571428571428571,   0.362902592520648},
                                                    { 0.285714285714286,  -0.571428571428571,  -0.857142857142857,   0.362902592520648},
                                                    { 0.571428571428571,  -0.857142857142857,  -0.857142857142857,   0.362902592520648},
                                                    {-0.857142857142857,  -0.857142857142857,   0.285714285714286,   0.362902592520648},
                                                    {-0.857142857142857,  -0.571428571428571,   0.000000000000000,   0.362902592520648},
                                                    {-0.857142857142857,  -0.285714285714286,  -0.285714285714286,   0.362902592520648},
                                                    {-0.857142857142857,   0.000000000000000,  -0.571428571428571,   0.362902592520648},
                                                    {-0.857142857142857,   0.285714285714286,  -0.857142857142857,   0.362902592520648},
                                                    {-0.571428571428571,  -0.857142857142857,   0.000000000000000,   0.362902592520648},
                                                    {-0.571428571428571,  -0.571428571428571,  -0.285714285714286,   0.362902592520648},
                                                    {-0.571428571428571,  -0.285714285714286,  -0.571428571428571,   0.362902592520648},
                                                    {-0.571428571428571,   0.000000000000000,  -0.857142857142857,   0.362902592520648},
                                                    {-0.285714285714286,  -0.857142857142857,  -0.285714285714286,   0.362902592520648},
                                                    {-0.285714285714286,  -0.571428571428571,  -0.571428571428571,   0.362902592520648},
                                                    {-0.285714285714286,  -0.285714285714286,  -0.857142857142857,   0.362902592520648},
                                                    { 0.000000000000000,  -0.857142857142857,  -0.571428571428571,   0.362902592520648},
                                                    { 0.000000000000000,  -0.571428571428571,  -0.857142857142857,   0.362902592520648},
                                                    { 0.285714285714286,  -0.857142857142857,  -0.857142857142857,   0.362902592520648},
                                                    {-0.857142857142857,  -0.857142857142857,   0.000000000000000,   0.362902592520648},
                                                    {-0.857142857142857,  -0.571428571428571,  -0.285714285714286,   0.362902592520648},
                                                    {-0.857142857142857,  -0.285714285714286,  -0.571428571428571,   0.362902592520648},
                                                    {-0.857142857142857,   0.000000000000000,  -0.857142857142857,   0.362902592520648},
                                                    {-0.571428571428571,  -0.857142857142857,  -0.285714285714286,   0.362902592520648},
                                                    {-0.571428571428571,  -0.571428571428571,  -0.571428571428571,   0.362902592520648},
                                                    {-0.571428571428571,  -0.285714285714286,  -0.857142857142857,   0.362902592520648},
                                                    {-0.285714285714286,  -0.857142857142857,  -0.571428571428571,   0.362902592520648},
                                                    {-0.285714285714286,  -0.571428571428571,  -0.857142857142857,   0.362902592520648},
                                                    { 0.000000000000000,  -0.857142857142857,  -0.857142857142857,   0.362902592520648},
                                                    {-0.857142857142857,  -0.857142857142857,  -0.285714285714286,   0.362902592520648},
                                                    {-0.857142857142857,  -0.571428571428571,  -0.571428571428571,   0.362902592520648},
                                                    {-0.857142857142857,  -0.285714285714286,  -0.857142857142857,   0.362902592520648},
                                                    {-0.571428571428571,  -0.857142857142857,  -0.571428571428571,   0.362902592520648},
                                                    {-0.571428571428571,  -0.571428571428571,  -0.857142857142857,   0.362902592520648},
                                                    {-0.285714285714286,  -0.857142857142857,  -0.857142857142857,   0.362902592520648},
                                                    {-0.857142857142857,  -0.857142857142857,  -0.571428571428571,   0.362902592520648},
                                                    {-0.857142857142857,  -0.571428571428571,  -0.857142857142857,   0.362902592520648},
                                                    {-0.571428571428571,  -0.857142857142857,  -0.857142857142857,   0.362902592520648},
                                                    {-0.857142857142857,  -0.857142857142857,  -0.857142857142857,   0.362902592520648},
                                                    {-0.833333333333333,  -0.833333333333333,   0.500000000000000,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.500000000000000,   0.166666666666667,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.166666666666667,  -0.166666666666667,  -0.932187812187812},
                                                    {-0.833333333333333,   0.166666666666667,  -0.500000000000000,  -0.932187812187812},
                                                    {-0.833333333333333,   0.500000000000000,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.500000000000000,  -0.833333333333333,   0.166666666666667,  -0.932187812187812},
                                                    {-0.500000000000000,  -0.500000000000000,  -0.166666666666667,  -0.932187812187812},
                                                    {-0.500000000000000,  -0.166666666666667,  -0.500000000000000,  -0.932187812187812},
                                                    {-0.500000000000000,   0.166666666666667,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.166666666666667,  -0.833333333333333,  -0.166666666666667,  -0.932187812187812},
                                                    {-0.166666666666667,  -0.500000000000000,  -0.500000000000000,  -0.932187812187812},
                                                    {-0.166666666666667,  -0.166666666666667,  -0.833333333333333,  -0.932187812187812},
                                                    { 0.166666666666667,  -0.833333333333333,  -0.500000000000000,  -0.932187812187812},
                                                    { 0.166666666666667,  -0.500000000000000,  -0.833333333333333,  -0.932187812187812},
                                                    { 0.500000000000000,  -0.833333333333333,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.833333333333333,   0.166666666666667,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.500000000000000,  -0.166666666666667,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.166666666666667,  -0.500000000000000,  -0.932187812187812},
                                                    {-0.833333333333333,   0.166666666666667,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.500000000000000,  -0.833333333333333,  -0.166666666666667,  -0.932187812187812},
                                                    {-0.500000000000000,  -0.500000000000000,  -0.500000000000000,  -0.932187812187812},
                                                    {-0.500000000000000,  -0.166666666666667,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.166666666666667,  -0.833333333333333,  -0.500000000000000,  -0.932187812187812},
                                                    {-0.166666666666667,  -0.500000000000000,  -0.833333333333333,  -0.932187812187812},
                                                    { 0.166666666666667,  -0.833333333333333,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.833333333333333,  -0.166666666666667,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.500000000000000,  -0.500000000000000,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.166666666666667,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.500000000000000,  -0.833333333333333,  -0.500000000000000,  -0.932187812187812},
                                                    {-0.500000000000000,  -0.500000000000000,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.166666666666667,  -0.833333333333333,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.833333333333333,  -0.500000000000000,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.500000000000000,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.500000000000000,  -0.833333333333333,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.833333333333333,  -0.833333333333333,  -0.833333333333333,  -0.932187812187812},
                                                    {-0.800000000000000,  -0.800000000000000,   0.400000000000000,   0.815498319838598},
                                                    {-0.800000000000000,  -0.400000000000000,   0.000000000000000,   0.815498319838598},
                                                    {-0.800000000000000,   0.000000000000000,  -0.400000000000000,   0.815498319838598},
                                                    {-0.800000000000000,   0.400000000000000,  -0.800000000000000,   0.815498319838598},
                                                    {-0.400000000000000,  -0.800000000000000,   0.000000000000000,   0.815498319838598},
                                                    {-0.400000000000000,  -0.400000000000000,  -0.400000000000000,   0.815498319838598},
                                                    {-0.400000000000000,   0.000000000000000,  -0.800000000000000,   0.815498319838598},
                                                    { 0.000000000000000,  -0.800000000000000,  -0.400000000000000,   0.815498319838598},
                                                    { 0.000000000000000,  -0.400000000000000,  -0.800000000000000,   0.815498319838598},
                                                    { 0.400000000000000,  -0.800000000000000,  -0.800000000000000,   0.815498319838598},
                                                    {-0.800000000000000,  -0.800000000000000,   0.000000000000000,   0.815498319838598},
                                                    {-0.800000000000000,  -0.400000000000000,  -0.400000000000000,   0.815498319838598},
                                                    {-0.800000000000000,   0.000000000000000,  -0.800000000000000,   0.815498319838598},
                                                    {-0.400000000000000,  -0.800000000000000,  -0.400000000000000,   0.815498319838598},
                                                    {-0.400000000000000,  -0.400000000000000,  -0.800000000000000,   0.815498319838598},
                                                    { 0.000000000000000,  -0.800000000000000,  -0.800000000000000,   0.815498319838598},
                                                    {-0.800000000000000,  -0.800000000000000,  -0.400000000000000,   0.815498319838598},
                                                    {-0.800000000000000,  -0.400000000000000,  -0.800000000000000,   0.815498319838598},
                                                    {-0.400000000000000,  -0.800000000000000,  -0.800000000000000,   0.815498319838598},
                                                    {-0.800000000000000,  -0.800000000000000,  -0.800000000000000,   0.815498319838598},
                                                    {-0.750000000000000,  -0.750000000000000,   0.250000000000000,  -0.280203089091978},
                                                    {-0.750000000000000,  -0.250000000000000,  -0.250000000000000,  -0.280203089091978},
                                                    {-0.750000000000000,   0.250000000000000,  -0.750000000000000,  -0.280203089091978},
                                                    {-0.250000000000000,  -0.750000000000000,  -0.250000000000000,  -0.280203089091978},
                                                    {-0.250000000000000,  -0.250000000000000,  -0.750000000000000,  -0.280203089091978},
                                                    { 0.250000000000000,  -0.750000000000000,  -0.750000000000000,  -0.280203089091978},
                                                    {-0.750000000000000,  -0.750000000000000,  -0.250000000000000,  -0.280203089091978},
                                                    {-0.750000000000000,  -0.250000000000000,  -0.750000000000000,  -0.280203089091978},
                                                    {-0.250000000000000,  -0.750000000000000,  -0.750000000000000,  -0.280203089091978},
                                                    {-0.750000000000000,  -0.750000000000000,  -0.750000000000000,  -0.280203089091978},
                                                    {-0.666666666666667,  -0.666666666666667,   0.000000000000000,   0.032544642857143},
                                                    {-0.666666666666667,   0.000000000000000,  -0.666666666666667,   0.032544642857143},
                                                    { 0.000000000000000,  -0.666666666666667,  -0.666666666666667,   0.032544642857143},
                                                    {-0.666666666666667,  -0.666666666666667,  -0.666666666666667,   0.032544642857143},
                                                    {-0.500000000000000,  -0.500000000000000,  -0.500000000000000,  -0.000752498530276}};

    const PetscReal nord12_3DGaussPoints[210][4] = {{-0.875000000000000,  -0.875000000000000,   0.625000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.625000000000000,   0.375000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.375000000000000,   0.125000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.125000000000000,  -0.125000000000000,  0.420407272132140},
                                                    {-0.875000000000000,   0.125000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.875000000000000,   0.375000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.875000000000000,   0.625000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.875000000000000,   0.375000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.625000000000000,   0.125000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.375000000000000,  -0.125000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.125000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.625000000000000,   0.125000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.625000000000000,   0.375000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.875000000000000,   0.125000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.625000000000000,  -0.125000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.375000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.125000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.375000000000000,   0.125000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.125000000000000,  -0.875000000000000,  -0.125000000000000,  0.420407272132140},
                                                    {-0.125000000000000,  -0.625000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.125000000000000,  -0.375000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.125000000000000,  -0.125000000000000,  -0.875000000000000,  0.420407272132140},
                                                    { 0.125000000000000,  -0.875000000000000,  -0.375000000000000,  0.420407272132140},
                                                    { 0.125000000000000,  -0.625000000000000,  -0.625000000000000,  0.420407272132140},
                                                    { 0.125000000000000,  -0.375000000000000,  -0.875000000000000,  0.420407272132140},
                                                    { 0.375000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140},
                                                    { 0.375000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140},
                                                    { 0.625000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.875000000000000,   0.375000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.625000000000000,   0.125000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.375000000000000,  -0.125000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.125000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.875000000000000,   0.125000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.875000000000000,   0.375000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.875000000000000,   0.125000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.625000000000000,  -0.125000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.375000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.125000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.625000000000000,   0.125000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.875000000000000,  -0.125000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.625000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.375000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.125000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.125000000000000,  -0.875000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.125000000000000,  -0.625000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.125000000000000,  -0.375000000000000,  -0.875000000000000,  0.420407272132140},
                                                    { 0.125000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140},
                                                    { 0.125000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140},
                                                    { 0.375000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.875000000000000,   0.125000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.625000000000000,  -0.125000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.375000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.125000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.875000000000000,   0.125000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.875000000000000,  -0.125000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.625000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.375000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.125000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.875000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.625000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.375000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.125000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.125000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140},
                                                    { 0.125000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.875000000000000,  -0.125000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.625000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.375000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.125000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.875000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.625000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.375000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.125000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.875000000000000,  -0.375000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.625000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.375000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.375000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.625000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.875000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140},
                                                    {-0.857142857142857,  -0.857142857142857,   0.571428571428571, -1.185481802234117},
                                                    {-0.857142857142857,  -0.571428571428571,   0.285714285714286, -1.185481802234117},
                                                    {-0.857142857142857,  -0.285714285714286,   0.000000000000000, -1.185481802234117},
                                                    {-0.857142857142857,   0.000000000000000,  -0.285714285714286, -1.185481802234117},
                                                    {-0.857142857142857,   0.285714285714286,  -0.571428571428571, -1.185481802234117},
                                                    {-0.857142857142857,   0.571428571428571,  -0.857142857142857, -1.185481802234117},
                                                    {-0.571428571428571,  -0.857142857142857,   0.285714285714286, -1.185481802234117},
                                                    {-0.571428571428571,  -0.571428571428571,   0.000000000000000, -1.185481802234117},
                                                    {-0.571428571428571,  -0.285714285714286,  -0.285714285714286, -1.185481802234117},
                                                    {-0.571428571428571,   0.000000000000000,  -0.571428571428571, -1.185481802234117},
                                                    {-0.571428571428571,   0.285714285714286,  -0.857142857142857, -1.185481802234117},
                                                    {-0.285714285714286,  -0.857142857142857,   0.000000000000000, -1.185481802234117},
                                                    {-0.285714285714286,  -0.571428571428571,  -0.285714285714286, -1.185481802234117},
                                                    {-0.285714285714286,  -0.285714285714286,  -0.571428571428571, -1.185481802234117},
                                                    {-0.285714285714286,   0.000000000000000,  -0.857142857142857, -1.185481802234117},
                                                    { 0.000000000000000,  -0.857142857142857,  -0.285714285714286, -1.185481802234117},
                                                    { 0.000000000000000,  -0.571428571428571,  -0.571428571428571, -1.185481802234117},
                                                    { 0.000000000000000,  -0.285714285714286,  -0.857142857142857, -1.185481802234117},
                                                    { 0.285714285714286,  -0.857142857142857,  -0.571428571428571, -1.185481802234117},
                                                    { 0.285714285714286,  -0.571428571428571,  -0.857142857142857, -1.185481802234117},
                                                    { 0.571428571428571,  -0.857142857142857,  -0.857142857142857, -1.185481802234117},
                                                    {-0.857142857142857,  -0.857142857142857,   0.285714285714286, -1.185481802234117},
                                                    {-0.857142857142857,  -0.571428571428571,   0.000000000000000, -1.185481802234117},
                                                    {-0.857142857142857,  -0.285714285714286,  -0.285714285714286, -1.185481802234117},
                                                    {-0.857142857142857,   0.000000000000000,  -0.571428571428571, -1.185481802234117},
                                                    {-0.857142857142857,   0.285714285714286,  -0.857142857142857, -1.185481802234117},
                                                    {-0.571428571428571,  -0.857142857142857,   0.000000000000000, -1.185481802234117},
                                                    {-0.571428571428571,  -0.571428571428571,  -0.285714285714286, -1.185481802234117},
                                                    {-0.571428571428571,  -0.285714285714286,  -0.571428571428571, -1.185481802234117},
                                                    {-0.571428571428571,   0.000000000000000,  -0.857142857142857, -1.185481802234117},
                                                    {-0.285714285714286,  -0.857142857142857,  -0.285714285714286, -1.185481802234117},
                                                    {-0.285714285714286,  -0.571428571428571,  -0.571428571428571, -1.185481802234117},
                                                    {-0.285714285714286,  -0.285714285714286,  -0.857142857142857, -1.185481802234117},
                                                    { 0.000000000000000,  -0.857142857142857,  -0.571428571428571, -1.185481802234117},
                                                    { 0.000000000000000,  -0.571428571428571,  -0.857142857142857, -1.185481802234117},
                                                    { 0.285714285714286,  -0.857142857142857,  -0.857142857142857, -1.185481802234117},
                                                    {-0.857142857142857,  -0.857142857142857,   0.000000000000000, -1.185481802234117},
                                                    {-0.857142857142857,  -0.571428571428571,  -0.285714285714286, -1.185481802234117},
                                                    {-0.857142857142857,  -0.285714285714286,  -0.571428571428571, -1.185481802234117},
                                                    {-0.857142857142857,   0.000000000000000,  -0.857142857142857, -1.185481802234117},
                                                    {-0.571428571428571,  -0.857142857142857,  -0.285714285714286, -1.185481802234117},
                                                    {-0.571428571428571,  -0.571428571428571,  -0.571428571428571, -1.185481802234117},
                                                    {-0.571428571428571,  -0.285714285714286,  -0.857142857142857, -1.185481802234117},
                                                    {-0.285714285714286,  -0.857142857142857,  -0.571428571428571, -1.185481802234117},
                                                    {-0.285714285714286,  -0.571428571428571,  -0.857142857142857, -1.185481802234117},
                                                    { 0.000000000000000,  -0.857142857142857,  -0.857142857142857, -1.185481802234117},
                                                    {-0.857142857142857,  -0.857142857142857,  -0.285714285714286, -1.185481802234117},
                                                    {-0.857142857142857,  -0.571428571428571,  -0.571428571428571, -1.185481802234117},
                                                    {-0.857142857142857,  -0.285714285714286,  -0.857142857142857, -1.185481802234117},
                                                    {-0.571428571428571,  -0.857142857142857,  -0.571428571428571, -1.185481802234117},
                                                    {-0.571428571428571,  -0.571428571428571,  -0.857142857142857, -1.185481802234117},
                                                    {-0.285714285714286,  -0.857142857142857,  -0.857142857142857, -1.185481802234117},
                                                    {-0.857142857142857,  -0.857142857142857,  -0.571428571428571, -1.185481802234117},
                                                    {-0.857142857142857,  -0.571428571428571,  -0.857142857142857, -1.185481802234117},
                                                    {-0.571428571428571,  -0.857142857142857,  -0.857142857142857, -1.185481802234117},
                                                    {-0.857142857142857,  -0.857142857142857,  -0.857142857142857, -1.185481802234117},
                                                    {-0.833333333333333,  -0.833333333333333,   0.500000000000000,  1.198527187098616},
                                                    {-0.833333333333333,  -0.500000000000000,   0.166666666666667,  1.198527187098616},
                                                    {-0.833333333333333,  -0.166666666666667,  -0.166666666666667,  1.198527187098616},
                                                    {-0.833333333333333,   0.166666666666667,  -0.500000000000000,  1.198527187098616},
                                                    {-0.833333333333333,   0.500000000000000,  -0.833333333333333,  1.198527187098616},
                                                    {-0.500000000000000,  -0.833333333333333,   0.166666666666667,  1.198527187098616},
                                                    {-0.500000000000000,  -0.500000000000000,  -0.166666666666667,  1.198527187098616},
                                                    {-0.500000000000000,  -0.166666666666667,  -0.500000000000000,  1.198527187098616},
                                                    {-0.500000000000000,   0.166666666666667,  -0.833333333333333,  1.198527187098616},
                                                    {-0.166666666666667,  -0.833333333333333,  -0.166666666666667,  1.198527187098616},
                                                    {-0.166666666666667,  -0.500000000000000,  -0.500000000000000,  1.198527187098616},
                                                    {-0.166666666666667,  -0.166666666666667,  -0.833333333333333,  1.198527187098616},
                                                    { 0.166666666666667,  -0.833333333333333,  -0.500000000000000,  1.198527187098616},
                                                    { 0.166666666666667,  -0.500000000000000,  -0.833333333333333,  1.198527187098616},
                                                    { 0.500000000000000,  -0.833333333333333,  -0.833333333333333,  1.198527187098616},
                                                    {-0.833333333333333,  -0.833333333333333,   0.166666666666667,  1.198527187098616},
                                                    {-0.833333333333333,  -0.500000000000000,  -0.166666666666667,  1.198527187098616},
                                                    {-0.833333333333333,  -0.166666666666667,  -0.500000000000000,  1.198527187098616},
                                                    {-0.833333333333333,   0.166666666666667,  -0.833333333333333,  1.198527187098616},
                                                    {-0.500000000000000,  -0.833333333333333,  -0.166666666666667,  1.198527187098616},
                                                    {-0.500000000000000,  -0.500000000000000,  -0.500000000000000,  1.198527187098616},
                                                    {-0.500000000000000,  -0.166666666666667,  -0.833333333333333,  1.198527187098616},
                                                    {-0.166666666666667,  -0.833333333333333,  -0.500000000000000,  1.198527187098616},
                                                    {-0.166666666666667,  -0.500000000000000,  -0.833333333333333,  1.198527187098616},
                                                    { 0.166666666666667,  -0.833333333333333,  -0.833333333333333,  1.198527187098616},
                                                    {-0.833333333333333,  -0.833333333333333,  -0.166666666666667,  1.198527187098616},
                                                    {-0.833333333333333,  -0.500000000000000,  -0.500000000000000,  1.198527187098616},
                                                    {-0.833333333333333,  -0.166666666666667,  -0.833333333333333,  1.198527187098616},
                                                    {-0.500000000000000,  -0.833333333333333,  -0.500000000000000,  1.198527187098616},
                                                    {-0.500000000000000,  -0.500000000000000,  -0.833333333333333,  1.198527187098616},
                                                    {-0.166666666666667,  -0.833333333333333,  -0.833333333333333,  1.198527187098616},
                                                    {-0.833333333333333,  -0.833333333333333,  -0.500000000000000,  1.198527187098616},
                                                    {-0.833333333333333,  -0.500000000000000,  -0.833333333333333,  1.198527187098616},
                                                    {-0.500000000000000,  -0.833333333333333,  -0.833333333333333,  1.198527187098616},
                                                    {-0.833333333333333,  -0.833333333333333,  -0.833333333333333,  1.198527187098616},
                                                    {-0.800000000000000,  -0.800000000000000,   0.400000000000000, -0.522755333229870},
                                                    {-0.800000000000000,  -0.400000000000000,   0.000000000000000, -0.522755333229870},
                                                    {-0.800000000000000,   0.000000000000000,  -0.400000000000000, -0.522755333229870},
                                                    {-0.800000000000000,   0.400000000000000,  -0.800000000000000, -0.522755333229870},
                                                    {-0.400000000000000,  -0.800000000000000,   0.000000000000000, -0.522755333229870},
                                                    {-0.400000000000000,  -0.400000000000000,  -0.400000000000000, -0.522755333229870},
                                                    {-0.400000000000000,   0.000000000000000,  -0.800000000000000, -0.522755333229870},
                                                    { 0.000000000000000,  -0.800000000000000,  -0.400000000000000, -0.522755333229870},
                                                    { 0.000000000000000,  -0.400000000000000,  -0.800000000000000, -0.522755333229870},
                                                    { 0.400000000000000,  -0.800000000000000,  -0.800000000000000, -0.522755333229870},
                                                    {-0.800000000000000,  -0.800000000000000,   0.000000000000000, -0.522755333229870},
                                                    {-0.800000000000000,  -0.400000000000000,  -0.400000000000000, -0.522755333229870},
                                                    {-0.800000000000000,   0.000000000000000 , -0.800000000000000, -0.522755333229870},
                                                    {-0.400000000000000,  -0.800000000000000,  -0.400000000000000, -0.522755333229870},
                                                    {-0.400000000000000,  -0.400000000000000,  -0.800000000000000, -0.522755333229870},
                                                    { 0.000000000000000,  -0.800000000000000 , -0.800000000000000, -0.522755333229870},
                                                    {-0.800000000000000,  -0.800000000000000,  -0.400000000000000, -0.522755333229870},
                                                    {-0.800000000000000,  -0.400000000000000,  -0.800000000000000, -0.522755333229870},
                                                    {-0.400000000000000,  -0.800000000000000,  -0.800000000000000, -0.522755333229870},
                                                    {-0.800000000000000,  -0.800000000000000,  -0.800000000000000, -0.522755333229870},
                                                    {-0.750000000000000,  -0.750000000000000,   0.250000000000000,  0.093401029697326},
                                                    {-0.750000000000000,  -0.250000000000000,  -0.250000000000000,  0.093401029697326},
                                                    {-0.750000000000000,   0.250000000000000,  -0.750000000000000,  0.093401029697326},
                                                    {-0.250000000000000,  -0.750000000000000,  -0.250000000000000,  0.093401029697326},
                                                    {-0.250000000000000,  -0.250000000000000,  -0.750000000000000,  0.093401029697326},
                                                    { 0.250000000000000,  -0.750000000000000 , -0.750000000000000,  0.093401029697326},
                                                    {-0.750000000000000,  -0.750000000000000,  -0.250000000000000,  0.093401029697326},
                                                    {-0.750000000000000,  -0.250000000000000,  -0.750000000000000,  0.093401029697326},
                                                    {-0.250000000000000,  -0.750000000000000,  -0.750000000000000,  0.093401029697326},
                                                    {-0.750000000000000,  -0.750000000000000,  -0.750000000000000,  0.093401029697326},
                                                    {-0.666666666666667,  -0.666666666666667,   0.000000000000000, -0.005325487012987},
                                                    {-0.666666666666667,   0.000000000000000,  -0.666666666666667, -0.005325487012987},
                                                    { 0.000000000000000,  -0.666666666666667,  -0.666666666666667, -0.005325487012987},
                                                    {-0.666666666666667,  -0.666666666666667,  -0.666666666666667, -0.005325487012987},
                                                    {-0.500000000000000,  -0.500000000000000,  -0.500000000000000,  0.000050166568685}};


    switch (numPoints)
    {
        case 1:
            PetscCall(renormalization3DGaussPoints(numPoints, nord1_3DGaussPoints, points, weights));
            break;
        case 4:
            PetscCall(renormalization3DGaussPoints(numPoints, nord2_3DGaussPoints, points, weights));
            break;
        case 5:
            PetscCall(renormalization3DGaussPoints(numPoints, nord3_3DGaussPoints, points, weights));
            break;
        case 11:
            PetscCall(renormalization3DGaussPoints(numPoints, nord4_3DGaussPoints, points, weights));
            break;
        case 14:
            PetscCall(renormalization3DGaussPoints(numPoints, nord5_3DGaussPoints, points, weights));
            break;
        case 24:
            PetscCall(renormalization3DGaussPoints(numPoints, nord6_3DGaussPoints, points, weights));
            break;
        case 31:
            PetscCall(renormalization3DGaussPoints(numPoints, nord7_3DGaussPoints, points, weights));
            break;
        case 43:
            PetscCall(renormalization3DGaussPoints(numPoints, nord8_3DGaussPoints, points, weights));
            break;
        case 53:
            PetscCall(renormalization3DGaussPoints(numPoints, nord9_3DGaussPoints, points, weights));
            break;
        case 126:
            PetscCall(renormalization3DGaussPoints(numPoints, nord10_3DGaussPoints, points, weights));
            break;
        case 210:
            PetscCall(renormalization3DGaussPoints(numPoints, nord12_3DGaussPoints, points, weights));
            break;
        default:
            break;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes the barycentric (affine) coordinates and their gradients on the reference tetrahedron.
 *
 * @param[in] X The point coordinates [xi, eta, zeta] in the reference tetrahedron.
 * @param[out] Lam The four barycentric coordinates [L0, L1, L2, L3].
 * @param[out] DLam The gradients of the barycentric coordinates, DLam[dim][coord_index].
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode AffineTetrahedron(PetscReal X[NUM_DIMENSIONS], PetscReal Lam[4], PetscReal DLam[NUM_DIMENSIONS][4]){
    
    PetscFunctionBeginUser;
    
    /* Define affine coordinates */
    Lam[0] = 1.-X[0]-X[1]-X[2];
    Lam[1] = X[0];
    Lam[2] = X[1];
    Lam[3] = X[2];
    
    /* Define gradients */
    DLam[0][0] = -1;
    DLam[0][1] =  1;
    DLam[1][0] = -1;
    DLam[1][2] =  1;
    DLam[2][0] = -1;
    DLam[2][3] =  1;

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Projects tetrahedral barycentric coordinates and gradients onto the 6 edges.
 *
 * @param[in] Lam The four barycentric coordinates [L0, L1, L2, L3].
 * @param[in] DLam The gradients of the barycentric coordinates.
 * @param[out] LampE Projections onto edges. LampE[edge_index][0/1] gives the two relevant barycentric coordinates for that edge.
 * @param[out] DLampE Projections of gradients onto edges. DLampE[edge_index][dim][0/1] gives the two relevant gradients.
 * @param[out] IdecE Boolean flag, always set to PETSC_FALSE as barycentric coordinates on an edge don’t sum to 1 unless the point is on the edge.
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode ProjectTetE(PetscReal Lam[4], PetscReal DLam[NUM_DIMENSIONS][4], PetscReal LampE[NUM_EDGES_PER_ELEMENT][2], PetscReal DLampE[NUM_EDGES_PER_ELEMENT][NUM_DIMENSIONS][2], PetscBool* IdecE){

    PetscFunctionBeginUser;
    
    /* Compute projection */
    
    /* e=0 --> edge10 with local orientation v1->v0 */
    PetscInt e = 0;
    LampE[e][0] = Lam[1];
    LampE[e][1] = Lam[0];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        DLampE[e][i][0] = DLam[i][1];
        DLampE[e][i][1] = DLam[i][0];
    }
    
    /* e=1 --> edge02 with local orientation v0->v2 */
    e = 1;
    LampE[e][0] = Lam[0];
    LampE[e][1] = Lam[2];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        DLampE[e][i][0] = DLam[i][0];
        DLampE[e][i][1] = DLam[i][2];
    }
    
    /* e=2 --> edge21 with local orientation v2->v1 */
    e = 2;
    LampE[e][0] = Lam[2];
    LampE[e][1] = Lam[1];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        DLampE[e][i][0] = DLam[i][2];
        DLampE[e][i][1] = DLam[i][1];
    }
    
    /* e=3 --> edge13 with local orientation v1->v3 */
    e = 3;
    LampE[e][0] = Lam[1];
    LampE[e][1] = Lam[3];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        DLampE[e][i][0] = DLam[i][1];
        DLampE[e][i][1] = DLam[i][3];
    }
    
    /* e=4 --> edge30 with local orientation v3->v0 */
    e = 4;
    LampE[e][0] = Lam[3];
    LampE[e][1] = Lam[0];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        DLampE[e][i][0] = DLam[i][3];
        DLampE[e][i][1] = DLam[i][0];
    }
    
    /* e=5 --> edge23 with local orientation v2->v3 */
    e = 5;
    LampE[e][0] = Lam[2];
    LampE[e][1] = Lam[3];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        DLampE[e][i][0] = DLam[i][2];
        DLampE[e][i][1] = DLam[i][3];
    }

    /* Projected coordinates are Lam, so IdecE=false for all edges */
    *IdecE = PETSC_FALSE;

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Orients edge-projected coordinates and gradients based on the edge orientation flag.
 * @param[in] S The pair of projected coordinates for the edge [s0, s1].
 * @param[in] DS The pair of projected gradients for the edge [Grad(s0), Grad(s1)].
 * @param[in] Nori The orientation flag (0 for original order, 1 for swapped order).
 * @param[out] GS The oriented coordinates [gs0, gs1].
 * @param[out] GDS The oriented gradients [Grad(gs0), Grad(gs1)].
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details If Nori is 0, GS = S and GDS = DS.
 *          If Nori is 1, GS = [s1, s0] and GDS = [Grad(s1), Grad(s0)].
 */

PetscErrorCode OrientE(PetscReal S[2], PetscReal DS[NUM_DIMENSIONS][2], PetscInt Nori, PetscReal GS[2], PetscReal GDS[NUM_DIMENSIONS][2]){

    PetscFunctionBeginUser;

    PetscInt Or[2][2];
    
    /* Nori=0 => (s0,s1)->(s0,s1) */
    Or[0][0] = 0;
    Or[0][1] = 1;

    /* Nori=1 => (s0,s1)->(s1,s0) */
    Or[1][0] = 1;
    Or[1][1] = 0;

    /* Local-to-global transformation */
    GS[0] = S[Or[Nori][0]];
    GS[1] = S[Or[Nori][1]];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        GDS[i][0] = DS[i][Or[Nori][0]];
        GDS[i][1] = DS[i][Or[Nori][1]];
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes shifted scaled Legendre polynomials P_i(y) where y = 2*X - T.
 *
 * Computes Legendre polynomials using the standard three-term recurrence relation,
 * adapted for the scaled variable y = 2*X - T.
 *
 * @param[in] X Coordinate, typically s1 from an oriented edge projection (range depends on T).
 * @param[in] T Scaling parameter, typically s0+s1 from an oriented edge projection.
 * @param[in] nord Maximum polynomial order required (computes P_0 to P_nord).
 * @param[out] P Array to store the computed polynomial values P[0] to P[nord].
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode PolyLegendre(PetscReal X, PetscReal T, PetscInt nord, PetscReal P[]){

    PetscFunctionBeginUser;

    /* Variables declaration */
    PetscReal y;

    /* i stands for the order of the polynomial, stored in P(i) lowest order case (order 0) */
    P[0] = 1.;
    
    /* First order case (order 1) if necessary */
    y = 2.0 * X - T;
    if(nord >= 1){
        P[1] = y;
    }
  
    if(nord >= 2){
        PetscReal tt = T * T;
        for(PetscInt i = 1; i < nord; i++){
            P[i + 1] = (2.0 * i + 1.0) * y * P[i] - i * tt * P[i-1];
            P[i + 1] /= (PetscReal)(i + 1.0);
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes Jacobi polynomials P_j^{alpha, 0} using a recurrence relation adapted for the shifted variable y = 2*X - T.
 *
 * @param[in] X Coordinate, typically s1 from an oriented face projection (range depends on T).
 * @param[in] T Scaling parameter, typically s0+s1 from an oriented face projection.
 * @param[in] nord Maximum polynomial order j required (0 to nord).
 * @param[in] Minalpha The starting value for alpha (alpha increases by 2 for different polynomial families).
 * @param[out] P Output 2D array P[family_index][order_j] storing the polynomial values.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details The output @p P stores polynomials for different alpha values (implicitly indexed by the first dimension,
 *          corresponding to Minalpha, Minalpha+2, …). Used for constructing face and volume basis functions.
 */
PetscErrorCode PolyJacobi(PetscReal X, PetscReal T, PetscInt nord, PetscInt Minalpha, PetscReal **P){
    
    PetscFunctionBeginUser;

    PetscReal *alpha, y;
    PetscInt minI = 0;
    PetscInt maxI = minI + nord;


    /* Allocate */
    PetscCall(PetscCalloc1(nord + 1, &alpha));

    /* Clearly (minI,maxI)=(0,nord), but the syntax is written as it is
       because it reflects how the indexing is called from outside */
    for(PetscInt i = 0; i < maxI + 1; i++){
        alpha[i] = Minalpha + 2*(i - minI);
    }
    
    /* Initiate first column (order 0) */
    for(PetscInt i = minI; i < maxI + 1; i++){
        P[i][0] = 1.;
    }
        
    /* Initiate second column (order 1) if necessary */
    y = 2*X - T;
    if(nord >= 1){
        for(PetscInt i = minI; i < maxI; i++){
            P[i][1] = y + alpha[i]*X;
        }
    }
    
    /* Fill the last columns if necessary */
    if(nord >= 2){
        PetscReal tt = pow(T, 2);
        PetscInt ni = -1;
        for(PetscInt i = 0; i < maxI - 1; i++){
            PetscReal al = alpha[i];
            PetscReal aa = pow(al, 2);
            ni += 1;
            /* Use recursion in order, i, to compute P^alpha_i for i>=2 */
            for(PetscInt j = 2; j < nord - ni + 1; j++){
                PetscReal ai = 2*j*(j+al)*(2*j+al-2);
                PetscReal bi = 2*j+al-1;
                PetscReal ci = (2*j+al)*(2*j+al-2);
                PetscReal di = 2*(j+al-1)*(j-1)*(2*j+al);
                
                P[i][j] = bi*(ci*y+aa*T)*P[i][j-1]-di*tt*P[i][j-2];
                P[i][j] = P[i][j]/ai;
            }
        }
    }

    /* Free memory */
    PetscCall(PetscFree(alpha));

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes homogenized Legendre polynomials L_i(s1 / (s0+s1)) * (s0+s1)^i.
 * @param[in] S Affine-like coordinates [s0, s1].
 * @param[in] nord Maximum polynomial order required (computes L_0 to L_nord).
 * @param[out] HomP Array to store the computed homogenized polynomial values HomP[0] to HomP[nord].
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Calls `PolyLegendre` with X = s1 and T = s0 + s1. The result `HomP[i]` corresponds
 *          to the i-th homogenized Legendre polynomial evaluated at S.
 */
PetscErrorCode HomLegendre(PetscReal S[2], PetscInt nord, PetscReal HomP[]){

    PetscFunctionBeginUser;

    PetscCall(PolyLegendre(S[1], S[0] + S[1], nord, HomP));

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes H(curl) ancillary basis functions associated with an edge.
 * @param[in] S Oriented edge coordinates [s0, s1].
 * @param[in] DS Oriented edge gradients [Grad(s0), Grad(s1)].
 * @param[in] nord Polynomial order for the element (determines number of edge functions).
 * @param[in] Idec Boolean flag (unused here, always PETSC_FALSE for edges).
 * @param[out] EE Output array (NUM_DIMENSIONS x nord) storing the vector value of each edge ancillary function.
 * @param[out] CurlEE Output array (NUM_DIMENSIONS x nord) storing the curl of each edge ancillary function.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Constructs the lowest-order Whitney edge function W = s0*Grad(s1) - s1*Grad(s0) and its curl Curl(W) = 2*Grad(s0) x Grad(s1).
 *          Computes higher-order functions by multiplying W by homogenized Legendre polynomials (from `HomLegendre`).
 *          The curl of the i-th function is (i+1)*P_{i-1}*Curl(W).
 */
PetscErrorCode AncEE(PetscReal S[2], PetscReal DS[NUM_DIMENSIONS][2], PetscInt nord, PetscBool Idec, PetscReal **EE, PetscReal **CurlEE){

    PetscFunctionBeginUser;

    // Local parameters
    PetscInt minI = 1;
    PetscInt maxI = nord;
    PetscInt Ncurl = 2*NUM_DIMENSIONS-3;
    
    PetscReal *homP;
    
    /* Allocate */
    PetscCall(PetscCalloc1(nord+1, &homP));
    
    /* Extract homogenized Legendre polyomials first */
    PetscCall(HomLegendre(S, maxI, homP));
    
    /* Simplified case */
    if(Idec){
        for(PetscInt i = minI; i < maxI + 1; i++){
            for(PetscInt j = 0; j < NUM_DIMENSIONS; j++){
                EE[j][i-1] = homP[i-1]*DS[j][1];
            }
        }
        /* No need to compute Whitney function or curl */
        for (PetscInt i = 0; i < Ncurl; ++i) {
            for (PetscInt j = minI - 1; j < maxI - 1; ++j) {
                CurlEE[i][j] = 0;
            }
        }
    } else {
        /* Lowest order Whitney function and its curl */
        PetscReal whiE[NUM_DIMENSIONS];
        PetscReal curlwhiE[NUM_DIMENSIONS];
        PetscReal temp1[NUM_DIMENSIONS], temp2[NUM_DIMENSIONS];

        for(PetscInt i = 0; i < NUM_DIMENSIONS; i++){
            whiE[i] = S[0]*DS[i][1] - S[1]*DS[i][0];
        }
        
        for(PetscInt i = 0; i < NUM_DIMENSIONS; i++){
            temp1[i] = DS[i][0];
            temp2[i] = DS[i][1];
        }
        PetscCall(crossProduct(temp1, temp2, curlwhiE));
        
        /* Now construct the higher order elements */
        for(PetscInt i = minI; i < maxI + 1; i++){
            for(PetscInt j = 0; j < NUM_DIMENSIONS; j++){
                EE[j][i-1] = homP[i-1]*whiE[j];
            }
            for(PetscInt j = 0; j < Ncurl; j++){
                CurlEE[j][i-1] = (i+1)*homP[i-1]*curlwhiE[j];
            }
        }
    }

    PetscCall(PetscFree(homP));   
    
    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Projects tetrahedral barycentric coordinates and gradients onto the 4 faces.
 * @param[in] Lam The four barycentric coordinates [L0, L1, L2, L3].
 * @param[in] DLam The gradients of the barycentric coordinates.
 * @param[out] LampF Projections onto faces. LampF[face_index][0/1/2] gives the three relevant barycentric coordinates.
 * @param[out] DLampF Projections of gradients onto faces. DLampF[face_index][dim][0/1/2] gives the three relevant gradients.
 * @param[out] IdecF Boolean flag, always set to PETSC_FALSE as barycentric coordinates on a face don't sum to 1 unless the point is on the face.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Maps the 4 barycentric coordinates/gradients to the triplet associated with each of the 4 faces according to a fixed local numbering convention (e.g., face 0 uses L1, L0, L2; face 1 uses L1, L3, L0, etc.).
 */
PetscErrorCode ProjectTetF(PetscReal Lam[4], PetscReal DLam[NUM_DIMENSIONS][4], PetscReal LampF[NUM_FACES_PER_ELEMENT][NUM_DIMENSIONS], PetscReal DLampF[NUM_FACES_PER_ELEMENT][NUM_DIMENSIONS][NUM_DIMENSIONS], PetscBool* IdecF){

    PetscFunctionBeginUser;
    
    /* Compute projection */
    
    /* f=0 --> face102 with local orientation v1->v0->v2 */
    PetscInt f = 0;
    LampF[f][0] = Lam[1];
    LampF[f][1] = Lam[0];
    LampF[f][2] = Lam[2];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        DLampF[f][i][0] = DLam[i][1];
        DLampF[f][i][1] = DLam[i][0];
        DLampF[f][i][2] = DLam[i][2];
    }
    
    /* f=1 --> face130 with local orientation v1->v3->v0 */
    f = 1;
    LampF[f][0] = Lam[1];
    LampF[f][1] = Lam[3];
    LampF[f][2] = Lam[0];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        DLampF[f][i][0] = DLam[i][1];
        DLampF[f][i][1] = DLam[i][3];
        DLampF[f][i][2] = DLam[i][0];
    }
    
    /* f=2 --> face123 with local orientation v1->v2->v3 */
    f = 2;
    LampF[f][0] = Lam[1];
    LampF[f][1] = Lam[2];
    LampF[f][2] = Lam[3];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        DLampF[f][i][0] = DLam[i][1];
        DLampF[f][i][1] = DLam[i][2];
        DLampF[f][i][2] = DLam[i][3];
    }
    
    /* f=3 --> face203 with local orientation v2->v0->v3 */
    f = 3;
    LampF[f][0] = Lam[2];
    LampF[f][1] = Lam[0];
    LampF[f][2] = Lam[3];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        DLampF[f][i][0] = DLam[i][2];
        DLampF[f][i][1] = DLam[i][0];
        DLampF[f][i][2] = DLam[i][3];
    }
    
    *IdecF = PETSC_FALSE;

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Orients face-projected coordinates and gradients based on the face orientation flag.
 * @param[in] S The triplet of projected coordinates for the face [s0, s1, s2].
 * @param[in] DS The triplet of projected gradients for the face [Grad(s0), Grad(s1), Grad(s2)].
 * @param[in] Nori The orientation flag (0-5, representing permutations of the vertices/coordinates).
 * @param[out] GS The oriented coordinates [gs0, gs1, gs2].
 * @param[out] GDS The oriented gradients [Grad(gs0), Grad(gs1), Grad(gs2)].
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Permutes the input coordinates and gradients based on the value of `Nori` (0-5), corresponding
 *          to the 6 possible orientations/permutations of the vertices of a triangle.
 */
PetscErrorCode OrientTri(PetscReal S[NUM_DIMENSIONS], PetscReal DS[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscInt Nori, PetscReal GS[NUM_DIMENSIONS], PetscReal GDS[NUM_DIMENSIONS][NUM_DIMENSIONS]){
    
    PetscFunctionBeginUser;

    PetscInt Or[NUM_DIMENSIONS*2][NUM_DIMENSIONS];
    
    /* Nori=0 => (s0,s1,s2)->(s0,s1,s2) */
    Or[0][0] = 0;
    Or[0][1] = 1;
    Or[0][2] = 2;
    
    /* Nori=1 => (s0,s1,s2)->(s1,s2,s0) */
    Or[1][0] = 1;
    Or[1][1] = 2;
    Or[1][2] = 0;

    /* Nori=2 => (s0,s1,s2)->(s2,s0,s1) */
    Or[2][0] = 2;
    Or[2][1] = 0;
    Or[2][2] = 1;

    /* Nori=3 => (s0,s1,s2)->(s0,s2,s1) */
    Or[3][0] = 0;
    Or[3][1] = 2;
    Or[3][2] = 1;

    /* Nori=4 => (s0,s1,s2)->(s1,s0,s2) */
    Or[4][0] = 1;
    Or[4][1] = 0;
    Or[4][2] = 2;

    /* Nori=5 => (s0,s1,s2)->(s2,s1,s0) */
    Or[5][0] = 2;
    Or[5][1] = 1;
    Or[5][2] = 0;
    
    /* Local-to-global transformation */
    GS[0] = S[Or[Nori][0]];
    GS[1] = S[Or[Nori][1]];
    GS[2] = S[Or[Nori][2]];
    
    for(PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        GDS[i][0] = DS[i][Or[Nori][0]];
        GDS[i][1] = DS[i][Or[Nori][1]];
        GDS[i][2] = DS[i][Or[Nori][2]];
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes integrated shifted scaled Jacobi polynomials L_j^{alpha,0} and related terms P, R.
 * @param[in] X Coordinate, typically s1 or s2 depending on context (e.g., face or volume basis).
 * @param[in] T Scaling parameter, typically s0+s1 or 1-sd depending on context.
 * @param[in] nord Maximum polynomial order required.
 * @param[in] Minalpha The starting value for alpha.
 * @param[in] Idec Boolean flag indicating if T=1 (simplified case).
 * @param[out] L Output 2D array storing the integrated polynomial values L_j^{alpha,0}.
 * @param[out] P Output 2D array storing the Jacobi polynomials P_j^{alpha,0} (derivative w.r.t. X).
 * @param[out] R Output 2D array storing terms related to the derivative w.r.t. T.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details First calls `PolyJacobi` to get the base polynomials P. Then uses recurrence relations
 *          involving P to compute the integrated polynomials L and the T-derivative related term R.
 *          Used for constructing face and volume basis functions.
 */
PetscErrorCode PolyIJacobi(PetscReal X, PetscReal T, PetscInt nord, PetscInt Minalpha, PetscBool Idec, PetscReal **L, PetscReal **P, PetscReal **R){

    PetscFunctionBeginUser;
    
    /* Clearly (minI,maxI)=(1,nord), but the syntax is written as it is
       because it reflects how the indexing is called from outside */
    PetscInt minI = 0;
    PetscInt maxI = minI + nord;
    PetscReal *alpha;
    PetscReal **ptemp;

    /* Allocate */
    PetscCall(PetscCalloc1(nord, &alpha));

    PetscCall(PetscCalloc1(nord + 1, &ptemp));
    for (PetscInt i = 0; i < nord + 1; i++){
        PetscCall(PetscCalloc1(nord + 1, &ptemp[i]));
    }
    
    PetscCall(PolyJacobi(X, T, nord, Minalpha, ptemp));
    
    /* Define P. Note that even though P is defined at all entries,
       because of the way Jacobi computes ptemp, only the necessary entries,
       and those on the first subdiagonal (which are never used later)
       are actually accurate.*/
    for(PetscInt i = minI; i < maxI; i++){
        for(PetscInt j = 0; j < nord; j++){
            P[i][j] = ptemp[i][j];
        }
    }

    /* Create vector alpha first */
    for(PetscInt i = 0; i < maxI; i++){
        alpha[i] = Minalpha + 2*(i - minI);
    }
    
    /* Initiate first column (order 1 in L) */
    for(PetscInt i = minI; i < maxI; i++){
        L[i][0] = X;
    }
    
    /* General case; compute R */
    for(PetscInt i = minI; i < maxI; i++){
        for(PetscInt j = 0; j < nord; j++){
            R[i][j] = 0;
        }
    }
    
    /* Fill the last columns if necessary */
    if(nord >= 2){
        PetscReal tt = pow(T, 2);
        PetscInt ni = -1;
      
        for(PetscInt i = 0; i < maxI - 1; i++){
            PetscReal al = alpha[i];
            ni += 1;
      
            for(PetscInt j = 2; j < nord - ni + 1; j++){
                PetscReal tia = j+j+al;
                PetscReal tiam1 = tia-1;
                PetscReal tiam2 = tia-2;
                PetscReal ai = (j+al)/(tiam1*tia);
                PetscReal bi = (al)/(tiam2*tia);
                PetscReal ci = (j-1)/(tiam2*tiam1);
      
                L[i][j-1] = ai*ptemp[i][j]+bi*T*ptemp[i][j-1]-ci*tt*ptemp[i][j-2];
                R[i][j-1] = -(j-1)*(ptemp[i][j-1]+T*ptemp[i][j-2]);
                R[i][j-1] = R[i][j-1]/tiam2;
            }
        }
    }
    
    /* Free memory */
    PetscCall(PetscFree(alpha));
    
    for (PetscInt i = 0; i < nord + 1; i++){
        PetscCall(PetscFree(ptemp[i]));
    }
    PetscCall(PetscFree(ptemp));

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes homogenized integrated Jacobi polynomials and their gradients.
 * @param[in] S Affine-like coordinates [s_a, s_b] (e.g., [s0+s1, s2] for faces).
 * @param[in] DS Gradients of S [Grad(s_a), Grad(s_b)].
 * @param[in] nord Maximum polynomial order required.
 * @param[in] Minalpha The starting value for alpha.
 * @param[in] Idec Boolean flag indicating if s_a + s_b = 1.
 * @param[out] HomL Output 2D array storing the homogenized integrated polynomial values.
 * @param[out] DHomL Output 3D array storing the gradients of HomL. DHomL[dim][alpha_idx][order_idx].
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Calls `PolyIJacobi` with appropriate arguments (X=s_b, T=s_a+s_b).
 *          Computes the gradient DHomL using the chain rule: DHomL = P * Grad(s_b) + R * Grad(s_a+s_b).
 *          Handles the simplified case where Idec=TRUE (T=1, so R term is not needed).
 */
PetscErrorCode HomIJacobi(PetscReal S[2], PetscReal DS[NUM_DIMENSIONS][2], PetscInt nord, PetscInt Minalpha, PetscBool Idec, PetscReal **HomL, PetscReal ***DHomL){

    PetscFunctionBeginUser;
    
    /* Clearly (minI,maxI)=(1,nord), but the syntax is written as it is
       because it reflects how the indexing is called from outside */
    PetscInt minI = 1;
    PetscInt maxI = minI+nord-1;
    
    PetscReal **homP;   /* homP[nord][nord] */
    PetscReal **homR;   /* homR[nord][nord] */

    /* Allocate */
    PetscCall(PetscCalloc1(nord, &homP));
    for (PetscInt i = 0; i < nord; i++){
        PetscCall(PetscCalloc1(nord, &homP[i]));
    }
     
    PetscCall(PetscCalloc1(nord, &homR));
    for (PetscInt i = 0; i < nord; i++){
        PetscCall(PetscCalloc1(nord, &homR[i]));
    }
        
    PetscInt ni = -1;
    
    if(Idec){
        PetscCall(PolyIJacobi(S[1], 1, nord, Minalpha, Idec, HomL, homP, homR));
        for(PetscInt i = minI; i < maxI + 1; i++){
            ni += 1;
            for(PetscInt j = 1; j < nord - ni + 1; j++){
                for(PetscInt k = 0; k < NUM_DIMENSIONS; k++){
                    DHomL[k][i-1][j-1] = homP[i-1][j-1] * DS[k][1];
                }
            }
        }
    } else {
        /* If sum of S different from 1 -> Idec=.FALSE. */
        
        PetscCall(PolyIJacobi(S[1], S[0] + S[1], nord, Minalpha, Idec, HomL, homP, homR));
        
        PetscReal DS01[NUM_DIMENSIONS];
        for(PetscInt i = 0; i < NUM_DIMENSIONS; i++){
            DS01[i] = DS[i][0] + DS[i][1];
        }
        
        for(PetscInt i = minI; i < maxI + 1; i++){
            ni += 1;
            for(PetscInt j = 1; j < nord - ni + 1; j++){
                for(PetscInt k = 0; k < NUM_DIMENSIONS; k++){
                    DHomL[k][i-1][j-1] = homP[i-1][j-1] * DS[k][1] + homR[i-1][j-1]*DS01[k];
                }
            }
        }    
    }
    
    /* Free memory */
    for (PetscInt i = 0; i < nord; i++){
        PetscCall(PetscFree(homP[i]));
    }
    PetscCall(PetscFree(homP));

    for (PetscInt i = 0; i < nord; i++){
        PetscCall(PetscFree(homR[i]));
    }
    PetscCall(PetscFree(homR));

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes H(curl) ancillary basis functions associated with a triangle face.
 * @param[in] S Oriented face coordinates [s0, s1, s2].
 * @param[in] DS Oriented face gradients [Grad(s0), Grad(s1), Grad(s2)].
 * @param[in] nord Polynomial order for the element.
 * @param[in] Idec Boolean flag (unused here).
 * @param[out] ETri Output 3D array (NUM_DIMENSIONS x nord-1 x nord-1) storing the vector value of each face ancillary function.
 * @param[out] CurlETri Output 3D array (NUM_DIMENSIONS x nord-1 x nord-1) storing the curl of each face ancillary function.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Constructs face functions by combining edge ancillary functions (`AncEE`) associated with the edge (s0, s1)
 *          and homogenized integrated Jacobi polynomials (`HomIJacobi`) depending on s2 and s0+s1.
 *          Calculates the curl using the product rule: Curl(EE * L) = Curl(EE)*L + Grad(L) x EE.
 *          The indices [j][k-1] correspond to polynomial orders related to the edge and the transverse direction.
 */
PetscErrorCode AncETri(PetscReal S[NUM_DIMENSIONS], PetscReal DS[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscInt nord, PetscBool Idec, PetscReal ***ETri, PetscReal ***CurlETri){

    PetscFunctionBeginUser;
    
    PetscReal DsL[NUM_DIMENSIONS][2];
    PetscReal sL[2];

    /* Local parameters */
    PetscInt minI = 0;
    PetscInt minJ = 1;
    PetscInt maxJ = nord-1;
    PetscInt maxIJ = nord-1;
    PetscInt minalpha = 2*minI+1;
    PetscInt Ncurl = 2*NUM_DIMENSIONS-3;
    PetscReal tempS[2] = {S[0], S[1]};
    PetscReal tempDS[NUM_DIMENSIONS][2];
    PetscReal **EE;         /* EE[NUM_DIMENSIONS][nord-minJ]*/
    PetscReal **curlEE;     /* curlEE[2*NUM_DIMENSIONS-3][nord-minJ] */
    PetscReal **homLal;     /* homLal[maxJ][maxJ] */
    PetscReal ***DhomLal;   /* DhomLal[NUM_DIMENSIONS][maxJ][maxJ] */
    PetscBool IdecE = PETSC_FALSE;

    /* get EE - this is never a simplified case (IdecE=0) */
    for(PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        tempDS[i][0] = DS[i][0];
        tempDS[i][1] = DS[i][1];
    }
    
    /* Allocate */
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &EE));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(nord-minJ, &EE[i]));
    }

    PetscCall(PetscCalloc1(2*NUM_DIMENSIONS-3, &curlEE));
    for (PetscInt i = 0; i < 2*NUM_DIMENSIONS-3; i++){
        PetscCall(PetscCalloc1(nord-minJ, &curlEE[i]));
    }

    PetscCall(PetscCalloc1(maxJ, &homLal));
    for (PetscInt i = 0; i < maxJ; i++){
        PetscCall(PetscCalloc1(maxJ, &homLal[i]));
    }

    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DhomLal));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(maxJ, &DhomLal[i]));
        for (PetscInt j = 0; j < maxJ; j++){
            PetscCall(PetscCalloc1(maxJ, &DhomLal[i][j]));
        }
    }

    /* Compute AncEE */
    PetscCall(AncEE(tempS, tempDS, nord-minJ, IdecE, EE, curlEE));
    
    /* Get homogenized Integrated Jacobi polynomials, homLal, and gradients */
    sL[0] = S[0]+S[1];
    sL[1] = S[2];
    for(PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        DsL[i][0] = DS[i][0] + DS[i][1];
        DsL[i][1] = DS[i][2];
    }
    
    /* Compute HomIJacobi */    
    PetscCall(HomIJacobi(sL, DsL, maxJ, minalpha, Idec, homLal, DhomLal));

    /* Simply complete the required information */
    for(PetscInt i = 0; i < maxIJ + 1; i++){
        for(PetscInt j = minI; j < i-minJ+1; j++){
            PetscInt k = i - j;

            for(PetscInt n = 0; n < NUM_DIMENSIONS; n++){
                ETri[n][j][k-1] = EE[n][j] * homLal[j][k-1];
            }
            
            PetscReal DhomLalxEE[NUM_DIMENSIONS];
            PetscReal temp1[NUM_DIMENSIONS], temp2[NUM_DIMENSIONS];
            
            for(PetscInt n = 0; n < NUM_DIMENSIONS; n++){
                temp1[n] = DhomLal[n][j][k-1];
                temp2[n] = EE[n][j];
            }
            
            PetscCall(crossProduct(temp1, temp2, DhomLalxEE));
            
            for(PetscInt n = 0; n < Ncurl; n++){
                CurlETri[n][j][k-1] = homLal[j][k-1]*curlEE[n][j] + DhomLalxEE[n];
            }
        }
    }
    
    /* Free memory */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscFree(EE[i]));   
    }
    PetscCall(PetscFree(EE));   

    for (PetscInt i = 0; i < 2*NUM_DIMENSIONS-3; i++){
        PetscCall(PetscFree(curlEE[i]));   
    }
    PetscCall(PetscFree(curlEE));   

    for (PetscInt i = 0; i < maxJ; i++){
        PetscCall(PetscFree(homLal[i]));   
    }
    PetscCall(PetscFree(homLal));

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        for (PetscInt j = 0; j < maxJ; j++){
            PetscCall(PetscFree(DhomLal[i][j]));   
            }
        PetscCall(PetscFree(DhomLal[i]));       
    }
    PetscCall(PetscFree(DhomLal));       

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes the orientation flags for the 4 faces and 6 edges of a tetrahedron cell.
 *
 * Uses DMPlexGetTransitiveClosure to get the cone of the cell (faces, edges, vertices) and their orientations.
 * Extracts the orientation flags provided by PETSc for faces and edges.
 * Translates the PETSc face orientation convention to the convention used in PETGEM.
 *
 * @param[in] dm The DMPlex object.
 * @param[in] cell The index of the cell.
 * @param[out] cellOrientation An array of size 10 to store the orientation flags [F0, F1, F2, F3, E0, E1, E2, E3, E4, E5].
 * @return PetscErrorCode PETSC_SUCCESS on success, or an error code otherwise.
 */
PetscErrorCode computeCellOrientation(DM dm, PetscInt cell, PetscInt cellOrientation[10]){
    
    PetscFunctionBeginUser;

    PetscInt transitiveClosureCellSize;
    PetscInt  *transitiveClosureCellPoints = NULL;
    PetscInt currentPoint; 
        
    /* Get transitive clousure/orientation for cell */
    /* Orden convention:
        - Faces indices start on position 2, orientation indices start on position 3
        - Edges indices start on position 2 + NUM_FACES_PER_ELEMENT * 2, orientation indices start on position 3 + NUM_FACES_PER_ELEMENT*2
        - Vertices indices start on position 2 + NUM_FACES_PER_ELEMENT * 2 + NUM_EDGES_PER_ELEMENT * 2

        Order convention for cellOrientation = F0, F1, F2, F3, E0, E1, E2, E3, E4, E5
    */
    PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &transitiveClosureCellSize, &transitiveClosureCellPoints));

    /* Get orientation for faces */
    currentPoint = 2;
    for (PetscInt i = 0; i < NUM_FACES_PER_ELEMENT; i++) {
        cellOrientation[i] = transitiveClosureCellPoints[currentPoint + 1];
        
        /* Cast to PETGEM basis functions orientation */
        switch (cellOrientation[i]){
            case -3:  cellOrientation[i] = 4; break;
            case -2:  cellOrientation[i] = 3; break;
            case -1:  cellOrientation[i] = 5; break;
            case  0:  cellOrientation[i] = 0; break;
            case  1:  cellOrientation[i] = 1; break;
            case  2:  cellOrientation[i] = 2; break;
            default: break;
        }
        currentPoint += 2;     
    }

    /* Get orientation for edges */
    currentPoint = 2 + NUM_FACES_PER_ELEMENT*2;     
    for (PetscInt i = 0; i < NUM_EDGES_PER_ELEMENT; i++) {
        cellOrientation[i+NUM_FACES_PER_ELEMENT] = transitiveClosureCellPoints[currentPoint + 1];
        
        /* Cast to PETGEM basis functions orientation */
        if (cellOrientation[i+NUM_FACES_PER_ELEMENT]<0){
            cellOrientation[i+NUM_FACES_PER_ELEMENT] = 1;
        }
        else{
            cellOrientation[i+NUM_FACES_PER_ELEMENT] = 0;
        }
        currentPoint += 2;
    }

    /* Restore transitive closure */
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &transitiveClosureCellSize, &transitiveClosureCellPoints));

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes the H(curl) conforming shape functions and their curls for a tetrahedron element.
 * @param[in] X Point coordinates [xi, eta, zeta] in the reference tetrahedron.
 * @param[in] nord Polynomial order.
 * @param[in] cellOrientation Array of 10 orientation flags (4 faces, 6 edges).
 * @param[out] ShapE Output array (NUM_DIMENSIONS x numDof) storing the vector value of each shape function.
 * @param[out] CurlE Output array (NUM_DIMENSIONS x numDof) storing the curl of each shape function.
 * @return PetscErrorCode PETSC_SUCCESS on success.
 * @details This function orchestrates the computation of all H(curl) shape functions (edge, face, volume)
 *          for a given order `nord` at a point `X`.
 *          1. Computes barycentric coordinates (`AffineTetrahedron`).
 *          2. Computes edge functions: Projects onto edges (`ProjectTetE`), orients (`OrientE`), computes ancillary edge functions (`AncEE`).
 *          3. Computes face functions: Projects onto faces (`ProjectTetF`), orients (`OrientTri`), handles 2 families per face, computes ancillary face functions (`AncETri`).
 *          4. Computes volume functions: Handles 3 families per cell, combines face ancillary functions (`AncETri` on sub-entities) with integrated Jacobi polynomials (`HomIJacobi` in the last barycentric coordinate).
 *          5. Reorders the computed functions from the hierarchical construction order to the PETSc DOF ordering convention.
 */
PetscErrorCode shape3DETet(PetscReal X[NUM_DIMENSIONS], PetscInt nord, PetscInt cellOrientation[10], PetscReal **ShapE, PetscReal **CurlE){

    PetscFunctionBeginUser;
   
    /* Local parameters */
    PetscInt numDofInCell;
    PetscBool IdecB[2] = {PETSC_FALSE, PETSC_FALSE};
    PetscInt minI = 0;
    PetscInt minJ = 1;
    PetscInt minK = 1;
    PetscInt minIJ = minI + minJ;
    PetscInt minIJK = minIJ + minK;
    PetscInt m = 0;     /* Initialize counter for shape functions */
    PetscInt NoriF[NUM_FACES_PER_ELEMENT];  /* Orientation for faces */
    PetscInt NoriE[NUM_EDGES_PER_ELEMENT];  /* Orientation for edges */
    
    PetscReal Lam[NUM_DIMENSIONS + 1] = {0.0};                      /* Define affine coordinates */
    PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1] = {{0.0}};   /* Define gradients */

    PetscBool IdecE;                                                
    PetscBool IdecF;                                                            /* Shape functions over faces */

    /* Compute number of dofs per cell */
    numDofInCell = nord * (nord + 2) * (nord + 3) / 2;

    /* Get affine tetrahedron */
    PetscCall(AffineTetrahedron(X, Lam, DLam));
    
    /* Extract orientation for faces */
    for (PetscInt i = 0; i < NUM_FACES_PER_ELEMENT; ++i){
        NoriF[i] = cellOrientation[i];
    }

    /* Extract orientation for edges */
    for (PetscInt i = 0; i < NUM_EDGES_PER_ELEMENT; ++i){
        NoriE[i] = cellOrientation[i+NUM_FACES_PER_ELEMENT]; 
    }

    /* Arrays for basis functions over edges */
    PetscReal LampE[NUM_EDGES_PER_ELEMENT][2];                      /* Shape functions over edges */
    PetscReal DLampE[NUM_EDGES_PER_ELEMENT][NUM_DIMENSIONS][2];     /* Shape functions over edges */
    PetscReal **EE;                                                 /* Shape functions over edges EE[NUM_DIMENSIONS][nord]; */
    PetscReal **CurlEE;                                             /* Curl of shape functions over edges CurlEE[2*NUM_DIMENSIONS-3][nord]; */   
    
    /* Allocate */
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &EE));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(nord, &EE[i]));
    }
    PetscCall(PetscCalloc1(2*NUM_DIMENSIONS-3, &CurlEE));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(nord, &CurlEE[i]));
    }    

    /* Reset matrices */
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++){
        for (PetscInt k = 0; k < numDofInCell; k++){
            ShapE[j][k] = 0.0;
            CurlE[j][k] = 0.0;
        }
    } 

    /* Compute edges projection */
    PetscCall(ProjectTetE(Lam, DLam, LampE, DLampE, &IdecE));

    /* Compute shape functions for edges */
    PetscInt nordEdge = 0;
    PetscInt numDofEdge = 0;    
    for(PetscInt i = 0; i < NUM_EDGES_PER_ELEMENT; i++){
        /* Local parameters */
        nordEdge = nord;
        numDofEdge = nordEdge;
        if(numDofEdge > 0){
            /* Local parameters */
            PetscInt maxI = nordEdge - 1;
            /* Orient */
            PetscReal GLampE[2] = {0.0};
            PetscReal GDLampE[3][2] = {{0.0}};
            PetscReal S[2] = {0.0};
            PetscReal D[NUM_DIMENSIONS][2] = {{0.0}};

            S[0] = LampE[i][0];
            S[1] = LampE[i][1];

            /* Extract the slice into D */
            for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
                for (PetscInt k = 0; k < 2; k++) {
                    D[j][k] = DLampE[i][j][k];
                }
            }

            /* Compute edge orientation */
            PetscCall(OrientE(S, D, NoriE[i], GLampE, GDLampE));

            /* Construct the shape functions */
            PetscCall(AncEE(GLampE, GDLampE, nordEdge, IdecE, EE, CurlEE));

            for(PetscInt j = minI; j < maxI + 1; j++){
                for(PetscInt k = 0; k < NUM_DIMENSIONS; k++){
                    ShapE[k][m] = EE[k][j];
                    CurlE[k][m] = CurlEE[k][j];
                }
                m += 1;
            }
        }
    }

    /* Free memory for shape functions on edges */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscFree(EE[i]));
    }
    PetscCall(PetscFree(EE));

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscFree(CurlEE[i]));
    }
    PetscCall(PetscFree(CurlEE));

    /* Arrays for basis functions over faces */
    PetscReal LampF[NUM_FACES_PER_ELEMENT][NUM_DIMENSIONS];                     /* Shape functions over faces */
    PetscReal DLampF[NUM_FACES_PER_ELEMENT][NUM_DIMENSIONS][NUM_DIMENSIONS];    /* Shape functions over faces */
    PetscReal ***ETri;                                                          /* Shape functions over faces ETri[NUM_DIMENSIONS][nord - 1][nord - 1] */
    PetscReal ***CurlETri;                                                      /* Shape functions over faces CurlETri[2*NUM_DIMENSIONS-3][nord-1][nord-1] */

    /* Allocate */
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &ETri));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(nord-1, &ETri[i]));
        for (PetscInt j = 0; j < nord-1; j++){
            PetscCall(PetscCalloc1(nord-1, &ETri[i][j]));
        }
    }

    PetscCall(PetscCalloc1(2*NUM_DIMENSIONS-3, &CurlETri));
    for (PetscInt i = 0; i < 2*NUM_DIMENSIONS-3; i++){
        PetscCall(PetscCalloc1(nord-1, &CurlETri[i]));
        for (PetscInt j = 0; j < nord-1; j++){
            PetscCall(PetscCalloc1(nord-1, &CurlETri[i][j]));
        }
    }

    /* Compute faces projection */
    PetscCall(ProjectTetF(Lam, DLam, LampF, DLampF, &IdecF));

    /* Compute shape functions for faces */
    PetscInt nordFace = 0;
    PetscInt numDofFace = 0;
    for(PetscInt i = 0; i < NUM_FACES_PER_ELEMENT; i++){
        /* Local parameters */
        nordFace = nord;
        numDofFace = nordFace*(nordFace-1)/2;
        
        if(numDofFace > 0){
            /* Local parameters (again) */
            PetscInt maxIJ = nordFace - 1;

            /* Orient */
            PetscReal GLampF[3];
            PetscReal GDLampF[3][3];
            PetscReal tmpLampF[3];
            PetscReal tempDLampF[3][3];

            /* Prepare input matrices */
            for (PetscInt j = 0; j<3; j++){
                tmpLampF[j] = LampF[i][j];
                for (PetscInt k = 0; k<3; k++){
                    tempDLampF[j][k] = DLampF[i][j][k];
                }
            }

            /* Compute faces orientatio */
            PetscCall(OrientTri(tmpLampF, tempDLampF, NoriF[i], GLampF, GDLampF));

            /* Loop over families */
            PetscInt famctr = m;
            for(PetscInt j = 0; j < 2; j++){
                m = famctr + j - 1;
                PetscInt abc[3];
                for(PetscInt k = 0; k < 3; k++){
                    PetscInt pos = (k - j) % 3;
                    if (pos < 0){
                        pos += 3;
                    } 
                    abc[pos] = k;
                }
        
                PetscReal tempGLampF[3];
                PetscReal tempGDLampF[3][3];
                for(PetscInt k = 0; k < 3; k++){
                    tempGLampF[k] = GLampF[abc[k]];
                    for(PetscInt t = 0; t < NUM_DIMENSIONS; t++){
                        tempGDLampF[t][k] = GDLampF[t][abc[k]];
                    }
                }

                /* Construct the shape functions */
                PetscCall(AncETri(tempGLampF, tempGDLampF, nordFace, IdecF, ETri, CurlETri));

                 for(PetscInt k = minIJ; k < maxIJ + 1; k++){
                    for(PetscInt r = minI; r < k-minJ+1; r++){
                        PetscInt p = k - r;
                        m += 2;
                        for(PetscInt t = 0; t < NUM_DIMENSIONS; t++){
                            ShapE[t][m-1] = ETri[t][r][p-1];
                            CurlE[t][m-1] = CurlETri[t][r][p-1];
                        }
                    }
                }
            }
        }
    }

    /* Free memory */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        for (PetscInt j = 0; j < nord-1; j++){
            PetscCall(PetscFree(ETri[i][j]));
        }
        PetscCall(PetscFree(ETri[i]));
    }
    PetscCall(PetscFree(ETri));   

    for (PetscInt i = 0; i < 2*NUM_DIMENSIONS-3; i++){
        for (PetscInt j = 0; j < nord-1; j++){
            PetscCall(PetscFree(CurlETri[i][j]));
        }
        PetscCall(PetscFree(CurlETri[i]));
    }
    PetscCall(PetscFree(CurlETri));

    /* Variables for basis functions in cell volume */
    PetscInt nordB = nord;
    PetscInt ndofB = nordB*(nordB-1)*(nordB-2)/6;
    PetscInt minbeta = 2*minIJ;
    PetscInt maxIJK = nordB-1;
    PetscInt maxK = maxIJK-minIJ;

    /* Arrays for basis functions in cell volume */
    PetscReal ***ETriV;         /* Shape functions in volume */
    PetscReal ***CurlETriV;     /* Shape functions in volume */
    PetscReal **homLbet;        /* Shape functions in volume */
    PetscReal ***DhomLbet;      /* Shape functions in volume */

    /* Allocate */
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &ETriV));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(nord-minK-1, &ETriV[i]));
        for (PetscInt j = 0; j < nord-minK-1; j++){
            PetscCall(PetscCalloc1(nord-minK-1, &ETriV[i][j]));
        }
    }

    PetscCall(PetscCalloc1(2*NUM_DIMENSIONS-3, &CurlETriV));
    for (PetscInt i = 0; i < 2*NUM_DIMENSIONS-3; i++){
        PetscCall(PetscCalloc1(nord-minK-1, &CurlETriV[i]));
        for (PetscInt j = 0; j < nord-minK-1; j++){
            PetscCall(PetscCalloc1(nord-minK-1, &CurlETriV[i][j]));
        }
    }

    PetscCall(PetscCalloc1(maxK, &homLbet));
    for (PetscInt i = 0; i < maxK; i++){
        PetscCall(PetscCalloc1(maxK, &homLbet[i]));
    }

    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DhomLbet));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(maxK, &DhomLbet[i]));
        for (PetscInt j = 0; j < maxK; j++){
            PetscCall(PetscCalloc1(maxK, &DhomLbet[i][j]));
        }
    }    

    /* If necessary, create bubbles */
    if(ndofB > 0){
        /* Local parameters (again) */
        IdecB[0] = IdecF;
        IdecB[1] = PETSC_TRUE;
        
        /* Loop over families */
        PetscInt famctr = m;
        for(PetscInt i = 0; i < 3; i++){
            m = famctr + i - 2;
            PetscInt abcd[4];
            for(PetscInt j = 0; j < 4; j++){
                PetscInt pos = (j - i) % 4;
                if (pos < 0){
                    pos += 4;
                } 
                abcd[pos] = j;
            }
            
            PetscInt abc[3] = {abcd[0], abcd[1], abcd[2]};
            PetscInt d = abcd[3];

            /* Now construct the shape functions (no need to orient) */
            PetscReal tempLam[3];
            PetscReal tempDLam[3][3];
            for(PetscInt j = 0; j < 3; j++){
                tempLam[j] = Lam[abc[j]];
                for(PetscInt k = 0; k < NUM_DIMENSIONS; k++){
                    tempDLam[k][j] = DLam[k][abc[j]];
                }
            }
            
            PetscCall(AncETri(tempLam, tempDLam, nordB-minK, IdecB[0], ETriV, CurlETriV));

            PetscReal tmp1[2] = {1-Lam[d], Lam[d]};
            PetscReal tmp2[3][2];

            /* Initialize input matrix */
            for(PetscInt j = 0; j < NUM_DIMENSIONS; j++){
                tmp2[j][0] = -DLam[j][d]; 
                tmp2[j][1] = DLam[j][d]; 
            }

            PetscCall(HomIJacobi(tmp1, tmp2, maxK, minbeta, IdecB[1], homLbet, DhomLbet));

            for(PetscInt j = minIJK; j < maxIJK+1; j++){
                for(PetscInt k = minIJ; k < j-minK+1; k++){
                    for(PetscInt r = minI; r < k-minJ+1; r++){
                        PetscInt p = k - r;
                        PetscInt q = j - k;
                        m += 3;

                        for(PetscInt n = 0; n < NUM_DIMENSIONS; n++){
                            ShapE[n][m-1] = ETriV[n][r][p-1]*homLbet[k-1][q-1];
                        }

                        PetscReal DhomLbetxETri[NUM_DIMENSIONS];
                        PetscReal v1[NUM_DIMENSIONS], v2[NUM_DIMENSIONS];
    
                        for(PetscInt n = 0; n < NUM_DIMENSIONS; n++){
                            v1[n] = DhomLbet[n][k-1][q-1];
                            v2[n] = ETriV[n][r][p-1];
                        }
    
                        PetscCall(crossProduct(v1, v2, DhomLbetxETri));
    
                        for(PetscInt n = 0; n < NUM_DIMENSIONS; n++){
                            CurlE[n][m-1] = homLbet[k-1][q-1]*CurlETriV[n][r][p-1] + DhomLbetxETri[n];
                        }
                    }
                }
            }
        }
    }

    /* Compute reordering vector (PETGEM to PETSc convention) */
    PetscInt tmp[] = {0,   1,  2,  3,  4,  5,                                                         /* p=1 (6 dofs per cell) */
                      12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, /* p=2 (20 dofs per cell) */
                      42, 43, 44, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, /* p=3 (45 dofs per cell) */
                      35, 36, 37, 38, 39, 40, 41,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 
                      13, 14, 15, 16, 17,
                      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 24, 25, 26, 27, 28, 29, 30, 31, /* p=4 (84 dofs per cell) */
                      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 
                      52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  
                       0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
                      20, 21, 22, 23,
                     110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, /* p=5 (140 dofs per cell) */
                     130, 131, 132, 133, 134, 135, 136, 137, 138, 139,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                      40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                      80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                     100, 101, 102, 103, 104, 105, 106, 107, 108, 109,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
                      10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,     
                     156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, /* p=6 (216 dofs per cell) */
                     176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
                     196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
                      36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
                      56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
                      76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
                      96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                     116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
                     136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
                       0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
                      20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35};

    PetscInt orderPermutation[numDofInCell];

    /* Define the starting indices of each tmp array */
    PetscInt offsets[] = {0, 6, 26, 71, 155, 295};

    for (PetscInt i = 0; i < numDofInCell; i++) {
        orderPermutation[i] = tmp[offsets[nord - 1] + i];
    }

    /* Copy basis and curl from PETGEM order convention */
    PetscReal tmpShapE[NUM_DIMENSIONS][numDofInCell], tmpCurlE[NUM_DIMENSIONS][numDofInCell]; 

    for (PetscInt i = 0; i<NUM_DIMENSIONS; i++){
        for (PetscInt j = 0; j<numDofInCell; j++){
            tmpShapE[i][j] = ShapE[i][j];
            tmpCurlE[i][j] = CurlE[i][j];        
        }    
    }

    /* Apply PETSc ordering */
    for (PetscInt i = 0; i<NUM_DIMENSIONS; i++){
        for (PetscInt j = 0; j<numDofInCell; j++){
            ShapE[i][j] = tmpShapE[i][orderPermutation[j]];
            CurlE[i][j] = tmpCurlE[i][orderPermutation[j]];
        }
    }

    /* Free memory */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        for (PetscInt j = 0; j < nordB-minK-1; j++){
            PetscCall(PetscFree(ETriV[i][j]));
        }
        PetscCall(PetscFree(ETriV[i]));
    }
    PetscCall(PetscFree(ETriV));   

    for (PetscInt i = 0; i < 2*NUM_DIMENSIONS-3; i++){
        for (PetscInt j = 0; j < nordB-minK-1; j++){
            PetscCall(PetscFree(CurlETriV[i][j]));
        }
        PetscCall(PetscFree(CurlETriV[i]));
    }
    PetscCall(PetscFree(CurlETriV));
    
    for (PetscInt i = 0; i < maxK; i++){
        PetscCall(PetscFree(homLbet[i]));   
    }
    PetscCall(PetscFree(homLbet));

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        for (PetscInt j = 0; j < maxK; j++){
            PetscCall(PetscFree(DhomLbet[i][j]));   
        }  
        PetscCall(PetscFree(DhomLbet[i]));       
    }
    PetscCall(PetscFree(DhomLbet));           

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes the elemental mass (Me) and stiffness (Ke) matrices for H(curl) elements.
 *
 * @param[in] nord Polynomial order.
 * @param[in] cellOrientation Array of 10 orientation flags (4 faces, 6 edges).
 * @param[in] jacobian The 3x3 Jacobian matrix of the element mapping.
 * @param[in] invJacobian The 3x3 inverse Jacobian matrix.
 * @param[in] numGaussPoints Number of Gauss quadrature points.
 * @param[in] gaussPoints Array of Gauss point coordinates (numGaussPoints x NUM_DIMENSIONS).
 * @param[in] weigths Array of Gauss point weights.
 * @param[in] cellResistivity Array containing the resistivity tensor components [rho_xx, rho_yy, rho_zz] (assumed diagonal).
 * @param[out] Me Output 2D array (numDof x numDof) for the elemental mass matrix.
 * @param[out] Ke Output 2D array (numDof x numDof) for the elemental stiffness matrix.
 * @return PetscErrorCode PETSC_SUCCESS on success, or an error code otherwise.
 */
PetscErrorCode computeElementalMatrix(PetscInt nord, PetscInt cellOrientation[10], PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscInt numGaussPoints, PetscReal **gaussPoints, PetscReal *weigths, PetscReal *cellResistivity, PetscReal **Me, PetscReal **Ke){
    PetscFunctionBeginUser;

    /* Initial declarations */
    PetscReal e_r[NUM_DIMENSIONS][NUM_DIMENSIONS]  = {{0.0}}; 
    PetscReal mu_r[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{0.0}}; 
    PetscReal iPoint[NUM_DIMENSIONS] = {0.0};
    PetscReal det;
    PetscReal temp1[NUM_DIMENSIONS];
    PetscReal temp2[NUM_DIMENSIONS];
    PetscReal temp3[NUM_DIMENSIONS];
    PetscReal temp4[NUM_DIMENSIONS];
    PetscReal dotResult; 
    PetscReal **ShapE;
    PetscReal **CurlE;
    PetscReal **NiReal;

    PetscInt numDofInCell; 

    /* Tensor for integration (Vertical transverse electric permitivity) */
    e_r[0][0] = cellResistivity[0];
    e_r[1][1] = cellResistivity[1];
    e_r[2][2] = cellResistivity[2];

    /* Tensor for integration (Constant magnetic permittivity) */
    mu_r[0][0] = 1.0;
    mu_r[1][1] = 1.0;
    mu_r[2][2] = 1.0;

    /* Compute the determinant of the Jacobian */ 
    det = jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1])
        - jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0])
        + jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]);

    /* Compute number of dofs per cell */
    numDofInCell = nord*(nord+2)*(nord+3)/2;    

    /* Allocate matrices for shape functions */    
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &ShapE));
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &CurlE));
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &NiReal));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(numDofInCell, &ShapE[i]));
        PetscCall(PetscCalloc1(numDofInCell, &CurlE[i]));
        PetscCall(PetscCalloc1(numDofInCell, &NiReal[i]));        
    }

    /* Reset elemental matrices */
    for (PetscInt i = 0; i < numDofInCell; ++i){
        for (PetscInt j = 0; j < numDofInCell; ++j){
                Me[i][j] = 0.0;
                Ke[i][j] = 0.0;
        }
    }

    /* Compute elemental matrices (mass and stifness matrix)*/    
    for (PetscInt i = 0; i < numGaussPoints; ++i){
        /* Get gauss for i point */
        iPoint[0] = gaussPoints[i][0];
        iPoint[1] = gaussPoints[i][1];
        iPoint[2] = gaussPoints[i][2];

        /* Compute basis function for i point */
        PetscCall(shape3DETet(iPoint, nord, cellOrientation, ShapE, CurlE));

        /* NiReal = Ni in real element */
        for (PetscInt j = 0; j < NUM_DIMENSIONS; ++j){
            for (PetscInt k = 0; k < numDofInCell; ++k){
                NiReal[j][k] = 0.0;
            }
        }

        for (PetscInt j = 0; j < NUM_DIMENSIONS; ++j){
            for (PetscInt k = 0; k < numDofInCell; ++k){
                for (PetscInt m = 0; m < NUM_DIMENSIONS; ++m){
                    NiReal[j][k] += (invJacobian[j][m] * ShapE[m][k]);
                }
            }
        }

        /* Perform mass matrix integration */
        for (PetscInt j = 0; j < numDofInCell; ++j){
            for (PetscInt k = 0; k < numDofInCell; ++k){
                /* Prepare data */
                for (PetscInt m = 0; m < NUM_DIMENSIONS; ++m){
                    /* Extract slide for row j */
                    temp1[m] = NiReal[m][j];
                    /* Extract slide for row k */
                    temp2[m] = NiReal[m][k];
                }

                /* Perform matrix vector multiplication */
                PetscCall(matrixVectorProduct(temp1, e_r, temp3));
                 
                /* Perform dot product */
                PetscCall(dotProduct(temp3, temp2, &dotResult));

                /* Integration */
                Me[j][k] += weigths[i] * dotResult * det;
            }
        }

        /* Transform curl on reference element to real element */
        for (PetscInt j = 0; j < numDofInCell; ++j){
            /* Prepare data */
            for (PetscInt k = 0; k < NUM_DIMENSIONS; ++k){
                /* Extract slide for row j */
                temp1[k] = CurlE[k][j];            
            }

            /* Perform vector matrix multiplication */
            PetscCall(vectorMatrixProduct(temp1, jacobian, temp2));

            /* Update data */
            for (PetscInt k = 0; k < NUM_DIMENSIONS; ++k){
                /* Update slide for row j */
                CurlE[k][j] = temp2[k] / det;
            }
        }

        /* Perform stiffness matrix integration */
        for (PetscInt j = 0; j < numDofInCell; ++j){
            for (PetscInt k = 0; k < numDofInCell; ++k){
                /* Prepare data */
                for (PetscInt m = 0; m < NUM_DIMENSIONS; ++m){
                    /* Extract slide for row j */
                    temp1[m] = CurlE[m][j];
                    temp2[m] = CurlE[m][k];                                        
                }

                /* Perform matrix vector multiplication */
                PetscCall(matrixVectorProduct(temp1, mu_r, temp3));
                 
                /* Perform point-wise multiplication */
                for (PetscInt m = 0; m < NUM_DIMENSIONS; ++m){
                    temp4[m] = temp2[m] * det;                 
                }

                /* Perform dot product */
                PetscCall(dotProduct(temp3, temp4, &dotResult));

                /* Integration */
                Ke[j][k] += weigths[i] * dotResult;
            }
        }
    }

    /* Free memory */ 
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscFree(ShapE[i]));
        PetscCall(PetscFree(CurlE[i]));
        PetscCall(PetscFree(NiReal[i]));
    }
    PetscCall(PetscFree(ShapE));
    PetscCall(PetscFree(CurlE));   
    PetscCall(PetscFree(NiReal));   

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes the H(curl) basis functions and their curls at a specific point in the reference element,
 *        transformed to the physical element.
 *
 * @param[in] nord Polynomial order.
 * @param[in] orientation Array of 10 orientation flags (4 faces, 6 edges).
 * @param[in] jacobian The 3x3 Jacobian matrix of the element mapping.
 * @param[in] invJacobian The 3x3 inverse Jacobian matrix.
 * @param[in] point The coordinates [xi, eta, zeta] in the reference element where functions are evaluated.
 * @param[out] basisFunctions Output array (NUM_DIMENSIONS x numDof) storing the vector value of each basis function in the physical element.
 * @param[out] curlBasisFunctions Output array (NUM_DIMENSIONS x numDof) storing the curl of each basis function in the physical element.
 * @return PetscErrorCode PETSC_SUCCESS on success, or an error code otherwise.
 */
PetscErrorCode computeBasisFunctions(PetscInt nord, PetscInt orientation[10], PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal *point, PetscReal **basisFunctions, PetscReal **curlBasisFunctions){
    PetscFunctionBeginUser;

    PetscReal iPoint[NUM_DIMENSIONS]  = {0.0};
    PetscReal temp1[NUM_DIMENSIONS];
    PetscReal temp2[NUM_DIMENSIONS];
    PetscReal det;
    PetscReal **ShapE;  /* Shape functions [NUM_DIMENSIONS][numDofInCell] */    
    PetscReal **CurlE;  /* Curl of shape functions [NUM_DIMENSIONS][numDofInCell] */

    PetscInt numDofInCell; 

    /* Compute number of dofs per cell */
    numDofInCell = nord*(nord+2)*(nord+3)/2;    

    /* Compute the determinant of the Jacobian */ 
    det = jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1])
        - jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0])
        + jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]);

    /* Allocate arrays */
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &ShapE));
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &CurlE));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(numDofInCell, &ShapE[i]));
        PetscCall(PetscCalloc1(numDofInCell, &CurlE[i]));
    }
    
    /* Initialize point coordinates */
    iPoint[0] = point[0];
    iPoint[1] = point[1];
    iPoint[2] = point[2];

    /* Compute basis function for iPoint */
    PetscCall(shape3DETet(iPoint, nord, orientation, ShapE, CurlE));    

    /* Reset basis functions and curl functions */
    for (PetscInt j = 0; j < NUM_DIMENSIONS; ++j){
        for (PetscInt k = 0; k < numDofInCell; ++k){
            basisFunctions[j][k] = 0.0;
            curlBasisFunctions[j][k] = 0.0;
        }
    }

    /* NiReal = Ni in real element */        
    for (PetscInt j = 0; j < NUM_DIMENSIONS; ++j){
        for (PetscInt k = 0; k < numDofInCell; ++k){
            for (PetscInt m = 0; m < NUM_DIMENSIONS; ++m){
                basisFunctions[j][k] += invJacobian[j][m] * ShapE[m][k];
            }
        }
    }

    /* Transform curl on reference element to real element */
    for (PetscInt i = 0; i < numDofInCell; ++i){
        /* Prepare data */
        for (PetscInt j = 0; j < NUM_DIMENSIONS; ++j){
            /* Extract slide for row j */
            temp1[j] = CurlE[j][i];            
        }

        /* Perform vector matrix multiplication */
        PetscCall(vectorMatrixProduct(temp1, jacobian, temp2));

        /* Update data */
        for (PetscInt j = 0; j < NUM_DIMENSIONS; ++j){
            /* Update slide for row j */
            curlBasisFunctions[j][i] = temp2[j] / det;
        }
    }

    /* Free memory */ 
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscFree(ShapE[i]));
        PetscCall(PetscFree(CurlE[i]));
    }
    PetscCall(PetscFree(ShapE));
    PetscCall(PetscFree(CurlE));   

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the elemental discrete gradient matrix for linear order 1 elements.
 *
 * @param[in] cellOrientation Array of 10 orientation flags, only edge orientations (indices 4-9) are used.
 * @param[out] gradientMatrix Output array (numEdges x numVertices = 6x4) storing the gradient matrix.
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode computeElementalGradientMatrix(PetscInt cellOrientation[10], PetscReal **gradientMatrix){
    PetscFunctionBeginUser;

    /* Variables declaration */ 
    PetscInt NoriE[NUM_EDGES_PER_ELEMENT];  /* Orientation for edges */

    /* Reset gradient matrix */
    for (PetscInt i = 0; i < NUM_EDGES_PER_ELEMENT; ++i){
        for (PetscInt j = 0; j < NUM_VERTICES_PER_ELEMENT; ++j){
                gradientMatrix[i][j] = 0;                
        }
    }

    /* Get matrix gradient directly from cellOrientation array 
       
       Order convention for cellOrientation = F0, F1, F2, F3, E0, E1, E2, E3, E4, E5 

       For gradients computation we consider:
         1 if this vextex is the edge ending point
        -1 if this vextex is the edge starting point
    */ 
    for (PetscInt i = 0; i < NUM_EDGES_PER_ELEMENT; ++i){
        NoriE[i] = cellOrientation[i+NUM_FACES_PER_ELEMENT]; 
    }

    for (PetscInt i = 0; i < NUM_EDGES_PER_ELEMENT; ++i){
        switch (i) {
        case 0: /* Edge 0: v0 --> v1 */  
            if (NoriE[i] == 0){
                gradientMatrix[i][0] =  1;
                gradientMatrix[i][1] = -1;
            }
            else{
                gradientMatrix[i][0] = -1;
                gradientMatrix[i][1] =  1;
            }
            break;
        case 1: /* Edge 1: v1 --> v2 */  
            if (NoriE[i] == 0){
                gradientMatrix[i][1] =  1;
                gradientMatrix[i][2] = -1;
            }
            else{
                gradientMatrix[i][1] = -1;
                gradientMatrix[i][2] =  1;
            }
            break;
        case 2: /* Edge 2: v2 --> v0 */  
            if (NoriE[i] == 0){
                gradientMatrix[i][2] =  1;
                gradientMatrix[i][0] = -1;
            }
            else{
                gradientMatrix[i][2] = -1;
                gradientMatrix[i][0] =  1;
            }
            break;
        case 3: /* Edge 3: v0 --> v3 */  
            if (NoriE[i] == 0){
                gradientMatrix[i][0] =  1;
                gradientMatrix[i][3] = -1;
            }
            else{
                gradientMatrix[i][0] = -1;
                gradientMatrix[i][3] =  1;
            }
            break;
        case 4: /* Edge 4: v3 --> v1 */  
            if (NoriE[i] == 0){
                gradientMatrix[i][3] =  1;
                gradientMatrix[i][1] = -1;
            }
            else{
                gradientMatrix[i][3] = -1;
                gradientMatrix[i][1] =  1;
            }
            break;
        case 5: /* Edge 5: v2 --> v3 */  
            if (NoriE[i] == 0){
                gradientMatrix[i][2] =  1;
                gradientMatrix[i][3] = -1;
            }
            else{
                gradientMatrix[i][2] = -1;
                gradientMatrix[i][3] =  1;
            }
            break;
        default: break;
        }    
    }
    
    PetscFunctionReturn(PETSC_SUCCESS);

}

/**
 * @brief Computes the standard H1 nodal basis functions (barycentric coordinates) and their gradients for a tetrahedron.
 * @param[in] Lam The four barycentric coordinates [L0, L1, L2, L3].
 * @param[in] DLam The gradients of the barycentric coordinates.
 * @param[out] LambV The values of the 4 nodal basis functions (LambV[i] = Lam[i]).
 * @param[out] DLambV The gradients of the 4 nodal basis functions (DLambV[i] = Grad(Lam[i])).
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details This function essentially just copies the barycentric coordinates and their gradients,
 *          as these are the standard P1 nodal basis functions on a tetrahedron.
 */

PetscErrorCode BlendTetV(PetscReal Lam[4], PetscReal DLam[NUM_DIMENSIONS][4], PetscReal LambV[NUM_VERTICES_PER_ELEMENT], PetscReal DLambV[NUM_VERTICES_PER_ELEMENT][NUM_DIMENSIONS]){
    /*Projection of tetrahedral edges in concordance with numbering of topological entities (vertices, edges, faces).

    :param ndarray Lam: affine coordinates
    :param ndarray DLam: gradients of affine coordinates
    :return: projection of affine coordinates on edges, projection of gradients of affine coordinates on edges
    :rtype: ndarray

    .. note:: References:\n
       Fuentes, F., Keith, B., Demkowicz, L., & Nagaraj, S. (2015). Orientation
       embedded high order shape functions for the exact sequence elements of
       all shapes. Computers & Mathematics with applications, 70(4), 353-458.
    */
    PetscFunctionBeginUser;

    /* Variable declaration */
    PetscInt    v; 

    /* 4 vertices, each with one blending function */

    /* v=1 --> v0=(0,0,0) */
    v = 0;
    LambV[v] = Lam[0]; 
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) 
        DLambV[v][i] = DLam[i][0];

    /* v=2 --> v1=(1,0,0) */
    v = 1;
    LambV[v] = Lam[1]; 
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) 
        DLambV[v][i] = DLam[i][1];

    /* v=3 --> v2=(0,1,0) */
    v = 2;
    LambV[v] = Lam[2]; 
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) 
        DLambV[v][i] = DLam[i][2];

    /* v=4 --> v3=(0,0,1) */
    v = 3;
    LambV[v] = Lam[3]; 
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) 
        DLambV[v][i] = DLam[i][3];

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes integrated shifted scaled Legendre polynomials L_i and related terms P, R.
 *
 * @param X [in] Coordinate, typically s1.
 * @param T [in] Scaling parameter, typically s0+s1.
 * @param nord [in] Maximum polynomial order required.
 * @param Idec [in] Boolean flag indicating if T=1 (simplified case).
 * @param homL [out] Output array storing the integrated polynomial values L_i.
 * @param homP [out] Output array storing the Legendre polynomials P_i (derivative w.r.t. X).
 * @param homR [out] Output array storing terms related to the derivative w.r.t. T.
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode PolyILegendre(PetscReal X, PetscReal T, PetscInt nord, PetscBool Idec, PetscReal homL[], PetscReal homP[], PetscReal homR[]){
    PetscFunctionBeginUser;

    /* Variables declaration */ 
    PetscReal *tmp;
    PetscReal tt, ifact;

    /* Allocate array */
    PetscCall(PetscCalloc1(nord+1, &tmp));
    
    /* Calling Legendre for required information */
    PetscCall(PolyLegendre(X, T, nord, tmp));
    
    for (PetscInt i = 1 ; i < nord; i++){
        homP[i-1] = tmp[i];
    }

    /* Integrated polynomial of order i is stored in L(i) */
    tt = T * T;

    /* Simplified case: no need to compute R */
    if (Idec) {
        for (PetscInt i = 1; i < nord; i++) {
            ifact = 4.0 * (i + 1) - 2.0;
            homL[i] = (tmp[i] - tt * tmp[i - 1]) / ifact;
        }
    } 
    else {
        for (PetscInt i = 1; i < nord; i++) {
            ifact = 4.0 * (i + 1) - 2.0;
            homL[i - 1] = (tmp[i + 1] - tt * tmp[i - 1]) / ifact;
            homR[i - 1] = -(tmp[i] + T * tmp[i - 1]) / 2;
        }
    }

    /* Free memory */
    PetscCall(PetscFree(tmp));

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes homogenized integrated Legendre polynomials and their gradients.
 *
 * @param[in] S Affine-like coordinates [s0, s1].
 * @param[in] DS Gradients of S [Grad(s0), Grad(s1)].
 * @param[in] nord Maximum polynomial order required.
 * @param[in] Idec Boolean flag indicating if s0 + s1 = 1.
 * @param[out] PhiE Output array storing the homogenized integrated polynomial values.
 * @param[out] DPhiE Output 2D array storing the gradients of PhiE. DPhiE[dim][order_idx].
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode HomILegendre(PetscReal S[2], PetscReal DS[NUM_DIMENSIONS][2], PetscInt nord, PetscBool Idec, PetscReal *PhiE, PetscReal **DPhiE){
    PetscFunctionBeginUser;

    /* Variable declaration */
    PetscReal *homL, *homP, *homR;
    PetscReal DS01[NUM_DIMENSIONS];

    /* Allocate arrays */
    PetscCall(PetscCalloc1(nord-1, &homL));
    PetscCall(PetscCalloc1(nord-1, &homP));
    PetscCall(PetscCalloc1(nord-1, &homR));
    
    /* Idec is the flag to compute x and t derivatives. If sum of S equal 1 -> Idec=TRUE */
    if (Idec){
        PetscCall(PolyILegendre(S[1], 1.0, nord, Idec, homL, homP, homR));
        for (PetscInt i = 1; i < nord; i++) {
            for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
                DPhiE[j][i] = homP[i - 1] * DS[j][0];
            }
        }
    }
    else {
        PetscCall(PolyILegendre(S[1], S[0] + S[1], nord, Idec, homL, homP, homR));
        for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
            DS01[i] = DS[i][0] + DS[i][1];
        }

        for (PetscInt i = 1; i < nord; i++) {
            for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
                DPhiE[j][i - 1] = homP[i - 1] * DS[j][1] + homR[i - 1] * DS01[j];
            }
        }
    }

    /* Prepare output for PhiE */
    for (PetscInt i = 0; i < nord - 1; i++){
        PhiE[i] = homL[i];
    }

    /* Free memory */
    PetscCall(PetscFree(homL));
    PetscCall(PetscFree(homP));
    PetscCall(PetscFree(homR));

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes H1 ancillary basis functions associated with an edge (integrated Legendre polynomials).
 * @param[in] S Oriented edge coordinates [s0, s1].
 * @param[in] DS Oriented edge gradients [Grad(s0), Grad(s1)].
 * @param[in] nord Polynomial order for the element.
 * @param[in] Idec Boolean flag indicating if s0 + s1 = 1.
 * @param[out] PhiE Output array storing the scalar value of each edge ancillary function.
 * @param[out] DPhiE Output 2D array storing the gradient of each edge ancillary function.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details This function simply calls `HomILegendre` as the required H1 edge functions are the
 *          homogenized integrated Legendre polynomials.
 */

PetscErrorCode AncPhiE(PetscReal S[2], PetscReal DS[NUM_DIMENSIONS][2], PetscInt nord, PetscBool Idec, PetscReal *PhiE, PetscReal **DPhiE){
    PetscFunctionBeginUser;

    /* These are precisely the homogenized Legendre polynomials */
    PetscCall(HomILegendre(S, DS, nord, Idec, PhiE, DPhiE));

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes H1 ancillary basis functions associated with a triangle face.
 *
 * @param[in] S Oriented face coordinates [s0, s1, s2].
 * @param[in] DS Oriented face gradients [Grad(s0), Grad(s1), Grad(s2)].
 * @param[in] nordFace Polynomial order for the element.
 * @param[in] IdecF Boolean flag (indicating if s0+s1+s2=1).
 * @param[out] PhiTri Output 2D array (nord-2 x nord-2) storing the scalar value of each face ancillary function.
 * @param[out] DPhiTri Output 3D array (NUM_DIMENSIONS x nord-2 x nord-2) storing the gradient of each face ancillary function.
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode AncPhiTri(PetscReal S[NUM_DIMENSIONS], PetscReal DS[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscInt nordFace, PetscBool IdecF, PetscReal **PhiTri, PetscReal ***DPhiTri){
    PetscFunctionBeginUser;

    /* Variables declaration */
    PetscInt minI = 2; 
    PetscInt minJ = 1; 
    PetscInt maxJ = nordFace-2;
    PetscInt minIJ = minI+minJ; 
    PetscInt maxIJ = nordFace;
    PetscInt minalpha = 2*minI;
    PetscBool IdecE = PETSC_FALSE;
    PetscReal GLampE[2] = {0.0};
    PetscReal GDLampE[NUM_DIMENSIONS][2] = {{0.0}};
    PetscReal *PhiE, **DPhiE;
    PetscReal DsL[NUM_DIMENSIONS][2];
    PetscReal sL[2]; 
    PetscReal **homLal;     /* homLal[maxJ][maxJ] */
    PetscReal ***DhomLal;   /* DhomLal[NUM_DIMENSIONS][maxJ][maxJ] */

    /* Allocate arrays */
    PetscCall(PetscCalloc1(nordFace-minJ-1, &PhiE));
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DPhiE));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(nordFace-minJ-1, &DPhiE[i]));
    }

    PetscCall(PetscCalloc1(maxJ, &homLal));
    for (PetscInt i = 0; i < maxJ; i++){
        PetscCall(PetscCalloc1(maxJ, &homLal[i]));
    }

    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DhomLal));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(maxJ, &DhomLal[i]));
        for (PetscInt j = 0; j < maxJ; j++){
            PetscCall(PetscCalloc1(maxJ, &DhomLal[i][j]));
        }
    }

    /* Prepare input */
    GLampE[0] = S[0];
    GLampE[1] = S[1];

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        GDLampE[i][0] = DS[i][0];  
        GDLampE[i][1] = DS[i][1];  
    }
    
    /* Get EE - this is never a simplified case (IdecE=0) */
    PetscCall(AncPhiE(GLampE, GDLampE, nordFace-minJ, IdecF, PhiE, DPhiE));

    /* Get homogenized Integrated Jacobi polynomials, homLal, and gradients */
    sL[0] = S[0]+S[1];
    sL[1] = S[2];
    for(PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        DsL[i][0] = DS[i][0] + DS[i][1];
        DsL[i][1] = DS[i][2];
    }

    /* Compute HomIJacobi */    
    PetscCall(HomIJacobi(sL, DsL, maxJ, minalpha, IdecF, homLal, DhomLal));

    /* Simply complete the required information */
    for (PetscInt i = minIJ; i <= maxIJ; i++) {
        for (PetscInt j = minI; j <= i - minJ; j++) {
            PetscInt k = i - j;
            PhiTri[j-2][k-1] = PhiE[j-2] * homLal[j-2][k-1];
            for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
                DPhiTri[l][j-2][k-1] = homLal[j-2][k-1] * DPhiE[l][j - 2] + PhiE[j - 2] * DhomLal[l][j - 2][k-1];
            }
        }
    }

    /* Free memory */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscFree(DPhiE[i]));
    }
    PetscCall(PetscFree(DPhiE));   
    PetscCall(PetscFree(PhiE));   

    for (PetscInt i = 0; i < maxJ; i++){
        PetscCall(PetscFree(homLal[i]));   
    }
    PetscCall(PetscFree(homLal));

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        for (PetscInt j = 0; j < maxJ; j++){
            PetscCall(PetscFree(DhomLal[i][j]));   
            }
        PetscCall(PetscFree(DhomLal[i]));       
    }
    PetscCall(PetscFree(DhomLal));


    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the H1 conforming shape functions and their gradients for a tetrahedron element.
 * @param[in] X Point coordinates [xi, eta, zeta] in the reference tetrahedron.
 * @param[in] nord Polynomial order.
 * @param[in] cellOrientation Array of 10 orientation flags (only edge/face orientations needed if nord > 1).
 * @param[out] ShapH Output array (numDof) storing the scalar value of each shape function.
 * @param[out] GradH Output array (NUM_DIMENSIONS x numDof) storing the gradient of each shape function.
 * @return PetscErrorCode PETSC_SUCCESS on success.
 * @details Computes hierarchical H1 basis functions (Lagrange):
 *          1. Vertex functions: Barycentric coordinates (`BlendTetV`).
 *          2. Edge functions (if nord > 1): Projects onto edges (`ProjectTetE`), orients (`OrientE`), computes ancillary edge functions (`AncPhiE`).
 *          3. Face functions (if nord > 2): Projects onto faces (`ProjectTetF`), orients (`OrientTri`), computes ancillary face functions (`AncPhiTri`).
 *          4. Volume functions (if nord > 3): Combines face ancillary functions (`AncPhiTri`) with integrated Jacobi polynomials (`HomIJacobi`).
 *          The functions are stored hierarchically (vertices first, then edges, faces, volume).
 * @note Reordering to PETSc convention is missing compared to `shape3DETet`. The implementation of volume functions seems incomplete/potentially incorrect in the provided snippet.
 */

PetscErrorCode shape3DHTet(PetscReal X[NUM_DIMENSIONS], PetscInt nord, PetscInt cellOrientation[10], PetscReal *ShapH, PetscReal **GradH){
    /*Compute values of 3D tetrahedron element H1 shape functions and their derivatives.

    :param ndarray X: master tetrahedron coordinates from (0,1)^3
    :param int nord: polynomial order
    :param ndarray NoriE: edge orientation
    :param ndarray NoriF: face orientation
    :return: number of dof, values of the shape functions at the point, curl of the shape functions
    :rtype: ndarray.

    .. note:: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D. (2016). Second-order 
       Nédélec curl-conforming prismatic element for computational electromagnetics. IEEE 
       Transactions on Antennas and Propagation, 64(10), 4384-4395.
    */
    PetscFunctionBeginUser;
   
    /* Local parameters */
    PetscBool IdecB[2] = {PETSC_FALSE, PETSC_FALSE};
    PetscInt minI = 1;
    PetscInt minJ = 0;
    PetscInt minK = 0;
    PetscInt minIJ = minI + minJ;
    PetscInt minIJK = minIJ + minK;
    PetscInt m = 0;     /* Initialize counter for shape functions */
    PetscInt NoriF[NUM_FACES_PER_ELEMENT];  /* Orientation for faces */
    PetscInt NoriE[NUM_EDGES_PER_ELEMENT];  /* Orientation for edges */
    
    PetscReal Lam[NUM_DIMENSIONS + 1] = {0.0};                      /* Define affine coordinates for tetra */
    PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1] = {{0.0}};   /* Define gradients for tetra */

    PetscReal LambV[NUM_VERTICES_PER_ELEMENT] = {0.0};                      /* Define affine coordinates for vertices */
    PetscReal DLambV[NUM_VERTICES_PER_ELEMENT][NUM_DIMENSIONS] = {{0.0}};   /* Define gradients for vertices */ 
  
    PetscInt MAXtetraH;
    MAXtetraH = ((nord+3)*(nord+2)*(nord+1))/6;

    PetscBool IdecE;                                                
    PetscBool IdecF;                                             /* Shape functions over faces */

    /* Reset matrices */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        ShapH[i] = 0.0;
        for (PetscInt j = 0; j < MAXtetraH; j++){
            GradH[i][j] = 0.0;
        }
    }

    /* Get affine tetrahedron */
    PetscCall(AffineTetrahedron(X, Lam, DLam));

    /* Define affine coordinates and gradients */
    PetscCall(BlendTetV(Lam, DLam, LambV, DLambV));
    for(PetscInt i = 0; i < NUM_VERTICES_PER_ELEMENT; i++){
        ShapH[m] = LambV[i];
        for(PetscInt j = 0; j < NUM_DIMENSIONS; j++){
            GradH[j][m] = DLambV[i][j];
        }
        m += 1;
    }

    /* Arrays for basis functions projection over edges */
    PetscReal LampE[NUM_EDGES_PER_ELEMENT][2];                      /* Shape functions over edges */
    PetscReal DLampE[NUM_EDGES_PER_ELEMENT][NUM_DIMENSIONS][2];     /* Shape functions over edges */

    /* Compute edges projection */
    PetscCall(ProjectTetE(Lam, DLam, LampE, DLampE, &IdecE));

    /* Extract orientation for faces */
    for (PetscInt i = 0; i < NUM_FACES_PER_ELEMENT; ++i){
        NoriF[i] = cellOrientation[i];
    }

    /* Extract orientation for edges */
    for (PetscInt i = 0; i < NUM_EDGES_PER_ELEMENT; ++i){
        NoriE[i] = cellOrientation[i+NUM_FACES_PER_ELEMENT]; 
    }

    /* Compute shape functions for edges */
    PetscInt nordEdge = 0;
    PetscInt numDofEdge = 0;    
    PetscReal *PhiE, **DPhiE;
    
    /* Allocate matrices for functions for edges */
    PetscCall(PetscCalloc1(nord-1, &PhiE));
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DPhiE));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(nord-1, &DPhiE[i]));
    }

    /* Loop over edges */
    for(PetscInt i = 0; i < NUM_EDGES_PER_ELEMENT; i++){
        /* Local parameters */
        nordEdge = nord;
        numDofEdge = nordEdge - 1;
        if(numDofEdge > 0){
            /* Local parameters */
            PetscInt maxI = nordEdge;
            /* Orient */
            PetscReal GLampE[2] = {0.0};
            PetscReal GDLampE[NUM_DIMENSIONS][2] = {{0.0}};
            PetscReal S[2] = {0.0};
            PetscReal D[NUM_DIMENSIONS][2] = {{0.0}};

            S[0] = LampE[i][0];
            S[1] = LampE[i][1];

            /* Extract the slice into D */
            for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
                for (PetscInt k = 0; k < 2; k++) {
                    D[j][k] = DLampE[i][j][k];
                }
            }

            /* Compute edge orientation */
            PetscCall(OrientE(S, D, NoriE[i], GLampE, GDLampE));

            /* Construct the shape functions */
            PetscCall(AncPhiE(GLampE, GDLampE, nordEdge, IdecE, PhiE, DPhiE));

            for(PetscInt j = minI; j < maxI; j++){
                ShapH[m] = PhiE[j-1];
                for(PetscInt k = 0; k < NUM_DIMENSIONS; k++){
                    GradH[k][m] = DPhiE[k][j-1];
                }
                m += 1;
            }
        }
    }

    /* Arrays for basis functions projection over faces */
    PetscReal LampF[NUM_FACES_PER_ELEMENT][NUM_DIMENSIONS];                     /* Shape functions over faces */
    PetscReal DLampF[NUM_FACES_PER_ELEMENT][NUM_DIMENSIONS][NUM_DIMENSIONS];    /* Shape functions over faces */
    PetscReal **PhiTri, ***DPhiTri; 

    /* Compute faces projection */
    PetscCall(ProjectTetF(Lam, DLam, LampF, DLampF, &IdecF));

    /* Allocate matrices for functions for faces */
    PetscCall(PetscCalloc1(nord-2, &PhiTri));
    for (PetscInt i = 0; i < nord-2; i++){
        PetscCall(PetscCalloc1(nord-2, &PhiTri[i]));
    }
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DPhiTri));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(nord-2, &DPhiTri[i]));
        for (PetscInt j = 0; j < nord-2; j++) {
            PetscCall(PetscCalloc1(nord-2, &DPhiTri[i][j]));
        }  
    }  

    /* Compute shape functions for faces */
    PetscInt nordFace = 0;
    PetscInt numDofFace = 0;
    for(PetscInt i = 0; i < NUM_FACES_PER_ELEMENT; i++){
        /* Local parameters */
        nordFace = nord;
        numDofFace = (nordFace-1)*(nordFace-2)/2;
        if(numDofFace > 0){
            /* Local parameters (again) */  
            PetscInt maxIJ = nordFace;  
            /* Orient */
            PetscReal GLampF[NUM_DIMENSIONS];
            PetscReal GDLampF[NUM_DIMENSIONS][NUM_DIMENSIONS];
            PetscReal tmpLampF[NUM_DIMENSIONS];
            PetscReal tempDLampF[NUM_DIMENSIONS][NUM_DIMENSIONS];

            /* Prepare input matrices */
            for (PetscInt j = 0; j<NUM_DIMENSIONS; j++){
                tmpLampF[j] = LampF[i][j];
                for (PetscInt k = 0; k<NUM_DIMENSIONS; k++){
                    tempDLampF[j][k] = DLampF[i][j][k];
                }
            }

            /* Compute faces orientation */
            PetscCall(OrientTri(tmpLampF, tempDLampF, NoriF[i], GLampF, GDLampF));

            /* Construct the shape functions */
            PetscCall(AncPhiTri(GLampF, GDLampF, nordFace, IdecF, PhiTri, DPhiTri));

            for(PetscInt j = minIJ+2; j <= maxIJ; j++){
                for(PetscInt k = minI+1; k < j-minJ; k++){
                    PetscInt l = j-k;
                    ShapH[m] = PhiTri[k-2][l-1];
                    for (PetscInt n = 0; n < NUM_DIMENSIONS; n++) {
                       GradH[n][m] = DPhiTri[n][k-2][l-1];
                    }
                    m += 1;
                }
            }
        }
    }

    /*PetscCall(PetscPrintf(PETSC_COMM_SELF, "ShapH\n" ));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "[" ));
    for (PetscInt i=0; i<m; i++){
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%f,\n", ShapH[i]));
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "];" ));

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n\n\nGradH\n" ));

    for (PetscInt i=0; i<NUM_DIMENSIONS; i++){
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "[" ));
        for (PetscInt j=0; j<m; j++){
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "%f\n", GradH[i][j]));
        }
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "];" ));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n" ));
    }
                
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n" ));*/



    /* If necessary, create bubbles (basis functions in volume)*/
    PetscInt nordB = nord;
    PetscInt ndofB = (nordB-1)*(nordB-2)*(nordB-3)/6;
    
    if(ndofB > 0){
        /* Local variables */
        PetscInt minbeta = 2*(minIJ+2);
        PetscInt maxIJK = nordB;
        PetscInt maxIJ = maxIJK-minK;
        PetscInt maxI = maxIJ-minJ;
        PetscInt maxJ = maxIJ-minI;
        PetscInt maxK = maxIJK-minIJ-2;
        PetscReal **PhiTriV, ***DPhiTriV; 
        PetscReal **homLbetV, ***DhomLbetV;
        PetscReal GLampV[NUM_DIMENSIONS];
        PetscReal GDLampV[NUM_DIMENSIONS][NUM_DIMENSIONS];

        IdecB[0] = IdecF;
        IdecB[1] = PETSC_TRUE;
        
        /* Allocate matrices for functions for volume */
        PetscCall(PetscCalloc1(nord-3, &PhiTriV));
        PetscCall(PetscCalloc1(nord-3, &homLbetV));
        for (PetscInt i = 0; i < nord-3; i++){
            PetscCall(PetscCalloc1(nord-3, &PhiTriV[i]));
            PetscCall(PetscCalloc1(nord-3, &homLbetV[i]));
        }
        PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DPhiTriV));
        PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DhomLbetV));
        for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
            PetscCall(PetscCalloc1(nord-3, &DPhiTriV[i]));
            PetscCall(PetscCalloc1(nord-3, &DhomLbetV[i]));
            for (PetscInt j = 0; j < nord-3; j++) {
                PetscCall(PetscCalloc1(nord-3, &DPhiTriV[i][j]));
                PetscCall(PetscCalloc1(nord-3, &DhomLbetV[i][j]));
            }  
        }  

        /* Prepare input matrices */
        for (PetscInt i = 0; i<NUM_DIMENSIONS; i++){
                GLampV[i] = Lam[i];
                for (PetscInt j = 0; j<NUM_DIMENSIONS; j++){
                    GDLampV[i][j] = DLam[i][j];
                }
        }

        /* Call phiTri and HomIJacobi - no need to orient */
        PetscCall(AncPhiTri(GLampV, GDLampV, nordB-minK-1, IdecB[0], PhiTriV, DPhiTriV));

        PetscReal tmp1[2] = {1-Lam[3], Lam[3]};
        PetscReal tmp2[NUM_DIMENSIONS][2];

        /* Initialize input matrix */
        for(PetscInt i = 0; i < NUM_DIMENSIONS; i++){
            tmp2[i][0] = -DLam[i][3]; 
            tmp2[i][1] = DLam[i][3]; 
        }

        PetscCall(HomIJacobi(tmp1, tmp2, maxK, minbeta, IdecB[1], homLbetV, DhomLbetV));


        
        for(PetscInt i = minIJK+3; i < maxIJK+1; i++){
            for(PetscInt j = minIJ; j < i-minK-2; j++){
                for(PetscInt k = minI; k < j-minJ+1; k++){
                    PetscInt p = j - k;
                    PetscInt q = i - j - 3;
                    ShapH[m] = PhiTriV[k-1][j]*homLbetV[j-1][k];
                    for (PetscInt n = 0; n < NUM_DIMENSIONS; n++) {
                        GradH[n][m] = homLbetV[j-1][k]*DPhiTriV[n][k-1][j] + PhiTriV[k-1][j]*DhomLbetV[n][j-1][k];
                    }                    
                    m+=1;        
                }
            }
        }
        
        /* Free memory */ 
        for (PetscInt i = 0; i < nord-3; i++) {
            PetscCall(PetscFree(PhiTriV[i]));
            PetscCall(PetscFree(homLbetV[i]));
        }
        PetscCall(PetscFree(PhiTriV));  
        PetscCall(PetscFree(homLbetV));  

        for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
            for (PetscInt j = 0; j < nord-3; j++) {
                PetscCall(PetscFree(DPhiTriV[i][j]));
                PetscCall(PetscFree(DhomLbetV[i][j]));
            }
            PetscCall(PetscFree(DPhiTriV[i]));
            PetscCall(PetscFree(DhomLbetV[i]));
        }
        PetscCall(PetscFree(DPhiTriV));
        PetscCall(PetscFree(DhomLbetV));


    
    }
    

    
    

    
    
    // PetscInt numDofInCell = nord * (nord + 2) * (nord + 3) / 2;
    // PetscInt orderPermutation[numDofInCell];

    // /* Define the starting indices of each tmp array */
    // PetscInt offsets[] = {0, 6, 26, 71, 155, 295};

    // for (PetscInt i = 0; i < numDofInCell; i++) {
    //     orderPermutation[i] = tmp[offsets[nord - 1] + i];
    // }

    // /* Copy basis and curl from PETGEM order convention */
    // PetscReal tmpShapE[NUM_DIMENSIONS][numDofInCell], tmpCurlE[NUM_DIMENSIONS][numDofInCell]; 

    // for (PetscInt i = 0; i<NUM_DIMENSIONS; i++){
    //     for (PetscInt j = 0; j<numDofInCell; j++){
    //         tmpShapE[i][j] = ShapE[i][j];
    //         tmpCurlE[i][j] = CurlE[i][j];        
    //     }    
    // }

    // /* Apply PETSc ordering */
    // for (PetscInt i = 0; i<NUM_DIMENSIONS; i++){
    //     for (PetscInt j = 0; j<numDofInCell; j++){
    //         ShapE[i][j] = tmpShapE[i][orderPermutation[j]];
    //         CurlE[i][j] = tmpCurlE[i][orderPermutation[j]];
    //     }
    // }

    // /* Free memory */
    // for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
    //     for (PetscInt j = 0; j < nordB-minK-1; j++){
    //         PetscCall(PetscFree(ETriV[i][j]));
    //     }
    //     PetscCall(PetscFree(ETriV[i]));
    // }
    // PetscCall(PetscFree(ETriV));   

    // for (PetscInt i = 0; i < 2*NUM_DIMENSIONS-3; i++){
    //     for (PetscInt j = 0; j < nordB-minK-1; j++){
    //         PetscCall(PetscFree(CurlETriV[i][j]));
    //     }
    //     PetscCall(PetscFree(CurlETriV[i]));
    // }
    // PetscCall(PetscFree(CurlETriV));
    
    // for (PetscInt i = 0; i < maxK; i++){
    //     PetscCall(PetscFree(homLbet[i]));   
    // }
    // PetscCall(PetscFree(homLbet));

    // for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
    //     for (PetscInt j = 0; j < maxK; j++){
    //         PetscCall(PetscFree(DhomLbet[i][j]));   
    //     }  
    //     PetscCall(PetscFree(DhomLbet[i]));       
    // }
    // PetscCall(PetscFree(DhomLbet));           


    /* Free memory */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscFree(DPhiE[i]));
    }
    PetscCall(PetscFree(DPhiE));   
    PetscCall(PetscFree(PhiE));   

    for (PetscInt i = 0; i < nord-2; i++) {
        PetscCall(PetscFree(PhiTri[i]));
    }
    PetscCall(PetscFree(PhiTri));  

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        for (PetscInt j = 0; j < nord-2; j++) {
            PetscCall(PetscFree(DPhiTri[i][j]));
        }
        PetscCall(PetscFree(DPhiTri[i]));
    }
    PetscCall(PetscFree(DPhiTri));


    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the elemental discrete gradient matrix (Placeholder/Incomplete for higher order).
 * @param[in] nord Polynomial order.
 * @param[in] cellOrientation Array of 10 orientation flags.
 * @param[in] numGaussPoints Number of Gauss quadrature points.
 * @param[in] gaussPoints Array of Gauss point coordinates.
 * @param[in] weigths Array of Gauss point weights.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details This function currently computes the H1 basis functions and gradients at the first Gauss point using `shape3DHTet`
 *          but doesn't actually compute or return the gradient matrix. Its purpose in the assembly context is unclear
 *          based on the implementation shown. It might be intended for testing or a different calculation.
 * @warning This function does not compute the discrete gradient matrix as its name suggests, especially for nord > 1.
 */
PetscErrorCode computeElementalGradientMatrix2(PetscInt nord, PetscInt cellOrientation[10], PetscInt numGaussPoints, PetscReal **gaussPoints, PetscReal *weigths){
    PetscFunctionBeginUser;

    /* Variables declaration */ 
    PetscInt MAXtetraH; 
    PetscReal *ShapH, **GradH;
    PetscReal iPoint[NUM_DIMENSIONS] = {0.0};

    MAXtetraH = ((nord+3)*(nord+2)*(nord+1))/6;


    /* Allocate matrices for shape functions */
    PetscCall(PetscCalloc1(MAXtetraH, &ShapH));
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &GradH));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(MAXtetraH, &GradH[i]));
    }

    /* Compute gradiend matrix */    
    for (PetscInt i = 0; i < 1; ++i){
        /* Get gauss for i point */
        iPoint[0] = gaussPoints[i][0];
        iPoint[1] = gaussPoints[i][1];
        iPoint[2] = gaussPoints[i][2]; 

        /* Compute basis function for i point */
        PetscCall(shape3DHTet(iPoint, nord, cellOrientation, ShapH, GradH));
    }

    
        
    PetscFunctionReturn(PETSC_SUCCESS);

}