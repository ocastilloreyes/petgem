/*
 * Filename: assembly.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2024-08-02
 *
 * Description:
 * This file contains functions for assembly linear system (CSEM or MT) in a PETGEM simulation. 
 *
*/

/* C libraries */ 


/* PETSc libraries */
#include <petscsys.h>

/* PETGEM functions */ 
#include "inputs.h"
#include "grid.h"
#include "transmitter.h"
#include "assembly.h"
#include "hvfem.h"
#include "constants.h"

/**
 * @brief Checks if the discrete gradient is in the kernel of the mass matrix M * G == 0.
 * @param[in] M Pointer to the mass matrix data (row-major).
 * @param[in] G Pointer to the discrete gradient matrix data (row-major).
 * @param[in] m Number of rows in M and G.
 * @param[in] n Number of columns in G (number of rows in H1 space).
 * @param[in] w Element identifier (for error reporting).
 * @return PetscErrorCode PETSC_SUCCESS on success.
 * @details This function verifies the property M * G = 0 for a given element's mass matrix (M)
 *          and discrete gradient matrix (G). It prints an error message if the product
 *          is not close to zero within PETSC_SMALL tolerance. The check is currently
 *          disabled by the `#if 0` block.
 */

PetscErrorCode check_kernel(PetscReal *M, PetscReal *G, PetscInt m, PetscInt n, PetscInt w)
{
   PetscFunctionBeginUser;
#if 0
   printf("DISC GRAD %d\n",w);
   for (PetscInt i = 0; i < m; i++) {
     for (PetscInt j = 0; j < n; j++) {
        printf("%g ",G[i*n + j]);
     }
     printf("\n");
   }
   printf("\n");
#endif
   for (PetscInt i = 0; i < m; i++) {
     for (PetscInt j = 0; j < n; j++) {
       PetscReal v = 0;
       for (PetscInt k = 0; k < m; k++) {
         // M is m x m, G is m x n
         v += M[i*m + k] * G[k * n + j];
       }
       if (!PetscIsCloseAtTol(v, 0, 0, PETSC_SMALL)) printf("Error element %d (%d %d)\n",w,i,j);
     }
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Assembles the linear system matrices (A, B, G) for the HVFEM formulation.
 *
 * This function orchestrates the assembly of the finite element system.
 * It computes elemental stiffness (K), mass (M), and discrete gradient (G) matrices
 * and assembles them into global PETSc Mat objects. It also computes the
 * right-hand side vector(s) based on the source definition (CSEM).
 * The final system matrix @p A is K - (i*omega*mu)*M.
 *
 * @param[in] dm The DMPlex object representing the mesh topology and H(curl) discretization.
 * @param[in] resistivity The Vec containing resistivity values, associated with a DM
 *                           providing cell-wise constants.
 * @param[in] grid A Grid struct containing mesh statistics and DOF information.
 * @param[in] sources A source struct containing source parameters (frequency, positions, etc.).
 * @param[in] params A Params struct containing simulation parameters (basis order, etc.).
 * @param[out] A Pointer to the assembled system matrix (K - i*omega*mu*M).
 * @param[out] B Pointer to the assembled right-hand side matrix (one column per source).
 * @param[out] G Pointer to the assembled discrete gradient matrix (maps H1 DOFs to H(curl) DOFs).
 * @return PetscErrorCode PETSC_SUCCESS on success, or an error code otherwise.
 */
PetscErrorCode assembleSystem(DM dm, Vec resistivity, Grid grid, setSource sources, Params params, Mat *A, Mat *B, Mat *G) 
{
   PetscFunctionBeginUser;
    
   /* Variables declaration */
   PetscInt  m, n, M, N, numGaussPoints, numCoords, numDofIndices, *dofIndices, numH1DofIndices, *H1dofIndices;
   PetscInt  cellOrientation[10], numResistivityComponents;
    
   PetscReal   cellResistivity[NUM_DIMENSIONS];   
   PetscReal   jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS];
   PetscReal   **gaussPoints, *weigths, **basisFunctions, **curlBasisFunctions, *XiEtaZeta;    
   PetscReal   **Me, **Ke, **gradientMatrix;
   PetscReal   omega;
   
   const PetscScalar   *arrayCoords;
   PetscScalar constFactor;
   PetscScalar *closureRHS;
   PetscScalar *closureLHS;
   PetscScalar *cellCoords = NULL;
   PetscScalar *resistivityValues = NULL;

   PetscBool   isDG, sourceType;
    
   PetscSection section, H1section;
    
   DM dmResistivity;

   MPI_Comm comm = PetscObjectComm((PetscObject)dm); 
   VecType vtype;
   Vec b, h1v;
   ISLocalToGlobalMapping mapping, H1mapping;
 
   /* Create linear system matrix */  
   PetscCall(DMSetAdjacency(dm, 0, PETSC_FALSE, PETSC_TRUE));
   PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));
   PetscCall(DMCreateMatrix(dm, A));
   PetscCall(MatSetFromOptions(*A));

   /* Create vector to store one right-hand side */
   PetscCall(DMCreateGlobalVector(dm, &b));
   PetscCall(DMGetLocalToGlobalMapping(dm, &mapping));
   PetscCall(VecSetLocalToGlobalMapping(b, mapping));
   PetscCall(VecSetOption(b, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));  
   PetscCall(VecSetFromOptions(b));

   /* Create discrete gradient matrix */
   PetscCall(DMCreateGlobalVector(grid.H1dm, &h1v));
   PetscCall(VecGetSize(b, &M));
   PetscCall(VecGetLocalSize(b, &m));
   PetscCall(VecGetSize(h1v, &N));
   PetscCall(VecGetLocalSize(h1v, &n));
   PetscCall(DMGetLocalToGlobalMapping(grid.H1dm, &H1mapping));
   PetscCall(MatCreate(comm, G));
   PetscCall(MatSetSizes(*G, m, n, M, N));
   PetscCall(MatSetType(*G, MATAIJ));
   PetscCall(MatSetLocalToGlobalMapping(*G, mapping, H1mapping));
   PetscCall(VecDestroy(&h1v));

   /* Create matrix to hold multiple right-hand sides */
   PetscCall(VecGetType(b, &vtype));
   PetscCall(MatCreateDenseFromVecType(comm, vtype, m, PETSC_DECIDE, M, sources.numSources, m, NULL, B));

   /* Print HEFEM statistics */
   PetscCall(PetscPrintf(comm, "\n HEFEM data:\n"));
   PetscCall(PetscPrintf(comm, "   Basis order             = %" PetscInt_FMT "\n", params.nord));
   PetscCall(PetscPrintf(comm, "   Num of dofs per vertex  = %" PetscInt_FMT "\n", grid.numDofInVertex));
   PetscCall(PetscPrintf(comm, "   Num of dofs per edge    = %" PetscInt_FMT "\n", grid.numDofInEdge));
   PetscCall(PetscPrintf(comm, "   Num of dofs per Face    = %" PetscInt_FMT "\n", grid.numDofInFace));
   PetscCall(PetscPrintf(comm, "   Num of dofs per volume  = %" PetscInt_FMT "\n", grid.numDofInVolume));
   PetscCall(PetscPrintf(comm, "   Num of dofs per cell    = %" PetscInt_FMT "\n", grid.numDofInCell));
   
   /* Print linear system statistics */
   PetscCall(PetscPrintf(comm, "\n Assembly linear system:\n"));
   PetscCall(PetscPrintf(comm, "   Num of MPI tasks    = %d\n", params.numMPITasks));
   PetscCall(PetscPrintf(comm, "   Vector size         = %" PetscInt_FMT "\n", M));
   PetscCall(PetscPrintf(comm, "   Matrix size         = %" PetscInt_FMT " x %" PetscInt_FMT "\n", M, M));
   PetscCall(PetscPrintf(comm, "   Assembly process    = Initiated\n"));

   /* Compute number of gauss points */
   PetscCall(computeNumGaussPoints3D(params.nord, &numGaussPoints));

   /* Allocate memory for gauss points */
   PetscCall(PetscCalloc1(numGaussPoints, &gaussPoints));
   for (PetscInt i = 0; i < numGaussPoints; i++) {
      PetscCall(PetscCalloc1(NUM_DIMENSIONS, &gaussPoints[i]));
   }
   PetscCall(PetscCalloc1(numGaussPoints, &weigths));

   /* Compute gauss points and its weigths */
   PetscCall(computeGaussPoints3D(numGaussPoints, gaussPoints, weigths));

   /* Allocate memory for RHS */
   PetscCall(PetscCalloc1(grid.numDofInCell, &closureRHS));
   PetscCall(PetscCalloc1(NUM_DIMENSIONS, &XiEtaZeta));
   PetscCall(PetscCalloc1(NUM_DIMENSIONS, &basisFunctions));
   PetscCall(PetscCalloc1(NUM_DIMENSIONS, &curlBasisFunctions));
   for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
      PetscCall(PetscCalloc1(grid.numDofInCell, &basisFunctions[i]));
      PetscCall(PetscCalloc1(grid.numDofInCell, &curlBasisFunctions[i]));
   }

   /* Get DM section */
   PetscCall(DMGetLocalSection(dm, &section));
   PetscCall(DMGetLocalSection(grid.H1dm, &H1section));

   /* Compute constant */
   omega = sources.freq * 2.0 * PETSC_PI;
   constFactor = (0.0 + 1.0*PETSC_i) * (omega * MU);      

   /* Check modeling mode */
   PetscCall(PetscStrcasecmp(params.mode, "CSEM", &sourceType));
   
   /* Perform finite element assembly for RHS (one vector per source) */    
   if (sourceType){  /* CSEM mode */
      /* Loop over number of sources */ 
      for (PetscInt i=0; i<sources.numSources; i++){
         /* Variables declaration */
         PetscReal sourceRotationVector[NUM_DIMENSIONS], sourceVector[NUM_DIMENSIONS];
         PetscReal Dx[NUM_DIMENSIONS] = {0.0};
         PetscReal Dy[NUM_DIMENSIONS] = {0.0};
         PetscReal Dz[NUM_DIMENSIONS] = {0.0};
         PetscInt sourceInCell; 

         PetscCall(VecZeroEntries(b));

         /* Define dipole for total electric field formulation */
         Dx[0] = sources.sourceArray[i].current * sources.sourceArray[i].length;   /* x-directed dipole */
         Dy[1] = sources.sourceArray[i].current * sources.sourceArray[i].length;   /* y-directed dipole */
         Dz[2] = sources.sourceArray[i].current * sources.sourceArray[i].length;   /* z-directed dipole */

         /* Compute matrices for source rotation */
         for (PetscInt j=0; j<NUM_DIMENSIONS; j++){
            sourceRotationVector[j] = 0.0;
         }
         PetscCall(vectorRotation(sources.sourceArray[i].azimuth, sources.sourceArray[i].dip, sourceRotationVector));

         /* Rotate source and setup electric field */
         sourceVector[0] = sourceRotationVector[0]*Dx[0] + sourceRotationVector[1]*Dy[0] + sourceRotationVector[2]*Dz[0];
         sourceVector[1] = sourceRotationVector[0]*Dx[1] + sourceRotationVector[1]*Dy[1] + sourceRotationVector[2]*Dz[1];
         sourceVector[2] = sourceRotationVector[0]*Dx[2] + sourceRotationVector[1]*Dy[2] + sourceRotationVector[2]*Dz[2];

         /* Locate source within computational domain */
         sourceInCell = -1; 
         PetscCall(locatePoint(dm, sources.sourceArray[i].position, &sourceInCell));

         /* Insert CSEM source */
         if (sourceInCell >= 0){
            /* Get cell coordinates */ 
            PetscCall(DMPlexGetCellCoordinates(dm, sourceInCell, &isDG, &numCoords, &arrayCoords, &cellCoords));

            /* Compute jacobian and its inverse for sourceInCell */
            PetscCall(computeJacobian(cellCoords, jacobian, invJacobian));
            
            /* Transform xyz source position to XiEtaZeta coordinates (reference tetrahedral element) */
            PetscCall(tetrahedronXYZToXiEtaZeta(cellCoords, sources.sourceArray[i].position, XiEtaZeta));
            
            /* Restore cell coordinates */ 
            PetscCall(DMPlexRestoreCellCoordinates(dm, sourceInCell, &isDG, &numCoords, &arrayCoords, &cellCoords));

            /* Compute cell orientation */
            for(PetscInt j = 0; j < 10; j++){
               cellOrientation[j] = 0;                
            }
            PetscCall(computeCellOrientation(dm, sourceInCell, cellOrientation));

            /* Compute basis functions for sourceInCell */ 
            PetscCall(computeBasisFunctions(params.nord, cellOrientation, jacobian, invJacobian, XiEtaZeta, basisFunctions, curlBasisFunctions));

            /* Get closure indices for sourceInCell */
            PetscCall(DMPlexGetClosureIndices(dm, section, section, sourceInCell, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));

            /* Compute contribution for closure */
            for (PetscInt j = 0; j < grid.numDofInCell; j++){
               closureRHS[j] = 0;
               for (PetscInt k = 0; k < NUM_DIMENSIONS; k++){
                  closureRHS[j] += (basisFunctions[k][j] * sourceVector[k]);
               }
            }

            /* Add closure to vector */
            PetscCall(VecSetValuesLocal(b, numDofIndices, dofIndices, closureRHS, INSERT_VALUES));

            /* Restore closure indices for sourceInCell */
            PetscCall(DMPlexRestoreClosureIndices(dm, section, section, sourceInCell, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
         }
         /* Perform global assembly for RHS */
         PetscCall(VecAssemblyBegin(b));
         PetscCall(VecAssemblyEnd(b));

         /* Copy rhs into B matrix */
         Vec bcol;
         PetscCall(MatDenseGetColumnVecWrite(*B, i, &bcol));
         PetscCall(VecCopy(b, bcol));
         PetscCall(MatDenseRestoreColumnVecWrite(*B, i, &bcol));
      }
   } else { /* MT mode */ 
   /* TODO */
   }
   PetscCall(VecDestroy(&b));

   /* Apply constant factor */
   PetscCall(MatScale(*B, constFactor));

   /* Allocate memory for LHS */
   PetscCall(PetscCalloc1(grid.numDofInCell*PetscMax(grid.numDofInCell, grid.numH1DofInCell), &closureLHS));
   PetscCall(PetscCalloc1(grid.numDofInCell, &Me));
   PetscCall(PetscCalloc1(grid.numDofInCell, &Ke));
   PetscCall(PetscCalloc1(grid.numDofInCell*grid.numDofInCell, &Me[0]));
   PetscCall(PetscCalloc1(grid.numDofInCell*grid.numDofInCell, &Ke[0]));
   for (PetscInt i = 1; i < grid.numDofInCell; i++){
      Me[i] = Me[i - 1] + grid.numDofInCell;
      Ke[i] = Ke[i - 1] + grid.numDofInCell;
   }
   PetscCall(PetscCalloc1(grid.numDofInCell, &gradientMatrix));
   PetscCall(PetscCalloc1(grid.numDofInCell * grid.numH1DofInCell, &gradientMatrix[0]));
   for (PetscInt i = 1; i < grid.numDofInCell; i++){
      gradientMatrix[i] = gradientMatrix[i - 1] + grid.numH1DofInCell;
   }
   
   /* Get the local values of the resistivity components */
   PetscCall(VecGetDM(resistivity, &dmResistivity));
   
   /* Perform finite element assembly for LHS */
   //for (PetscInt i = grid.cellStart; i < grid.cellEnd; ++i) {
   for (PetscInt i = grid.cellStart; i < 1; ++i) {   

      /* Get spatial coordinates for cell i */ 
      PetscCall(DMPlexGetCellCoordinates(dm, i, &isDG, &numCoords, &arrayCoords, &cellCoords));

      /* Compute jacobian and its inverse for cell i */
      PetscCall(computeJacobian(cellCoords, jacobian, invJacobian));



      PetscCall(PetscPrintf(comm, "\n"));
         PetscCall(PetscPrintf(comm, "Coordinates\n"));
         for(PetscInt j = 0; j < 4; j++){
            PetscCall(PetscPrintf(comm, "(%f, %f, %f)\n", PetscRealPart(cellCoords[j+0]), PetscRealPart(cellCoords[j+1]), PetscRealPart(cellCoords[j+2])));
         }




      /* Restore coordinates for cell i */ 
      PetscCall(DMPlexRestoreCellCoordinates(dm, i, &isDG, &numCoords, &arrayCoords, &cellCoords));

      /* Compute cell orientation */
      for(PetscInt j = 0; j < 10; j++){
         cellOrientation[j] = 0;
      }
      PetscCall(computeCellOrientation(dm, i, cellOrientation));

      /* Get resistivity for cell i */ 
      PetscCall(DMPlexVecGetClosure(dmResistivity, NULL, resistivity, i, &numResistivityComponents, &resistivityValues));
      PetscCheck(numResistivityComponents == 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "Exiting: found resistivity components != 3.\n");
      cellResistivity[0] = PetscRealPart(resistivityValues[0]);
      cellResistivity[1] = PetscRealPart(resistivityValues[1]);
      cellResistivity[2] = PetscRealPart(resistivityValues[2]);
      PetscCall(DMPlexVecRestoreClosure(dmResistivity, NULL, resistivity, i, &numResistivityComponents, &resistivityValues));

      /* Compute mass and stifness matrices for cell i */
      //PetscCall(computeElementalMatrix(params.nord, cellOrientation, jacobian, invJacobian, numGaussPoints, gaussPoints, weigths, cellResistivity, Me, Ke));
      
      /* Compute gradient matrix XXX TODO higher order*/
      //if (params.nord == 2) {
         
         /*PetscCall(PetscPrintf(comm, "Cell cellOrientation\n"));
         for(PetscInt j = 0; j < 10; j++){
            PetscCall(PetscPrintf(comm, "%d, \n", cellOrientation[j]));
         }
         PetscCall(PetscPrintf(comm, "\n"));
         PetscCall(PetscPrintf(comm, "Jacobian\n"));
         for(PetscInt j = 0; j < NUM_DIMENSIONS; j++){
            PetscCall(PetscPrintf(comm, "%f, %f, %f\n", jacobian[j][0], jacobian[j][2], jacobian[j][2]));
         }

         PetscCall(PetscPrintf(comm, "\n"));

         PetscCall(PetscPrintf(comm, "\n"));
         PetscCall(PetscPrintf(comm, "Inv jacobian\n"));
         for(PetscInt j = 0; j < NUM_DIMENSIONS; j++){
            PetscCall(PetscPrintf(comm, "%f, %f, %f\n", invJacobian[j][0], invJacobian[j][2], invJacobian[j][2]));
         }*/
         
        PetscCall(computeElementalGradientMatrix(cellOrientation, gradientMatrix));
        PetscCall(computeElementalGradientMatrix2(params.nord, cellOrientation, numGaussPoints, gaussPoints, weigths));

        // Check that gradientMatrix is the kernel of Ke
        //check_kernel(Ke[0], gradientMatrix[0], grid.numDofInCell, grid.numH1DofInCell, i);
      //}

      /* Get closure indices for cell i */
      PetscCall(DMPlexGetClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
      PetscCall(DMPlexGetClosureIndices(grid.H1dm, H1section, H1section, i, PETSC_TRUE, &numH1DofIndices, &H1dofIndices, NULL, NULL));

      /* Compute elemental matrix for cell i */
      PetscCall(PetscArrayzero(closureLHS, grid.numDofInCell * grid.numDofInCell));
      for(PetscInt j = 0; j < grid.numDofInCell; j++){
         for(PetscInt k = 0; k < grid.numDofInCell; k++){
            closureLHS[j * grid.numDofInCell + k] = Ke[j][k] - (constFactor * Me[j][k]);
         }
      }

      /* Add closure to matrix */
      PetscCall(MatSetValuesLocal(*A, numDofIndices, dofIndices, numDofIndices, dofIndices, closureLHS, ADD_VALUES));

      /* Insert closure to discrete gradient matrix */
      /* XXX TODO higher order*/
      if (params.nord == 1) {
        PetscCall(PetscArrayzero(closureLHS, grid.numDofInCell * grid.numH1DofInCell));
        for(PetscInt j = 0; j < grid.numDofInCell; j++){
           for(PetscInt k = 0; k < grid.numH1DofInCell; k++){
              closureLHS[j * grid.numH1DofInCell + k] = gradientMatrix[j][k];
           }
        }

        PetscCall(MatSetValuesLocal(*G, numDofIndices, dofIndices, numH1DofIndices, H1dofIndices, closureLHS, INSERT_VALUES));
      }

      /* Restore closure indices for cell i */
      PetscCall(DMPlexRestoreClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
      PetscCall(DMPlexRestoreClosureIndices(grid.H1dm, H1section, H1section, i, PETSC_TRUE, &numH1DofIndices, &H1dofIndices, NULL, NULL));
   }

   /* Perform global assembly for LHS */
   PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY));

   /* The discrete gradient matrix is used to compute mesh connectivity
      information within the solver. Just use nonzero dofs */
   /* XXX TODO higher order*/
   if (params.nord == 1) {
     PetscCall(MatFilter(*G, 0, PETSC_TRUE, PETSC_FALSE));
   }
   // If you set constFactor to 0 you can check that the error must be zero
#if 0
   {
   Mat T;
   PetscReal err;
   PetscCall(MatMatMult(*A, *G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T));
   PetscCall(MatNorm(T, NORM_FROBENIUS, &err));
   PetscCall(PetscPrintf(comm, "   Error Gradient %g\n", err));
   }
#endif

   /* End of assembly */
   PetscCall(PetscPrintf(comm, "   Assembly process    = Finished\n"));

   PetscCall(MatViewFromOptions(*A, NULL, "-petgem_mat_view"));
   PetscCall(MatViewFromOptions(*G, NULL, "-petgem_grad_view"));

   /* Free memory */
   PetscCall(PetscFree(weigths));
   for (PetscInt i = 0; i < numGaussPoints; i++) {
      PetscCall(PetscFree(gaussPoints[i]));
   }
   PetscCall(PetscFree(gaussPoints));

   PetscCall(PetscFree(Me[0]));
   PetscCall(PetscFree(Ke[0]));
   PetscCall(PetscFree(gradientMatrix[0]));
   PetscCall(PetscFree(gradientMatrix));
   PetscCall(PetscFree(Me));    
   PetscCall(PetscFree(Ke));
   PetscCall(PetscFree(closureLHS));

   for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
      PetscCall(PetscFree(basisFunctions[i]));
      PetscCall(PetscFree(curlBasisFunctions[i]));
   }
   PetscCall(PetscFree(basisFunctions));    
   PetscCall(PetscFree(curlBasisFunctions));
   PetscCall(PetscFree(XiEtaZeta));
   PetscCall(PetscFree(closureRHS));
   PetscFunctionReturn(PETSC_SUCCESS);
}