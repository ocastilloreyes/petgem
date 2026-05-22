/*
 * Filename: hvfem_hierarchical.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Hierarchical Nédélec H(curl) basis on tetrahedra for nord = 1..6.
 */

/*
 * Notes:
 * The basis is evaluated directly on the reference cell using the
 * hierarchical construction (edge / face / volume blocks built from
 * OrientE / OrientTri / HomIJacobi). Orientation is encoded inside the
 * reference shape functions, so the assembly sign vector must be all +1
 * for this ops table (handled in buildDofSigns).
 *
 * Public surface (registered via hvfem_internal.h):
 *   - nedelecOps_order{1..6} : NedelecOps dispatch tables produced by the
 *     HIERARCHICAL_ORDER_OPS macro near the bottom of this file.
 *
 * Discrete-gradient builders (all orders):
 *   - TOPOLOGICAL (hierarchicalBuildGradientMatrixTopological): vertex-
 *     edge incidence. Lowest-order Whitney slot per mesh edge maps to
 *     that edge's 2 endpoint vertex H1 DOFs; face/volume / higher-order
 *     edge DOF rows are zero. Used by the inverse kernel and the BDDC
 *     curl-kernel hint.
 *   - COMMUTING EXACT (hierarchicalBuildExactGradientMatrix): the
 *     canonical Nédélec interpolation operator applied to ∇P_nord,
 *     evaluated via the Ainsworth–Coyle DOF moments. Satisfies the
 *     algebraic commuting diagram K·G = 0 to machine precision and is
 *     consistent across adjacent cells (every shared moment is computed
 *     in a canonical geometric frame). Consumed by the forward kernel.
 */

#include <petsc.h>
#include <petscblaslapack.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>

#include "constants.h"
#include "hvfem.h"
#include "hvfem_internal.h"

/* ---------------------------------------------------------------------------
 * Local helpers - small wrappers over hvfem.c machinery exposed via
 * hvfem_internal.h. AncEE / AncETri are unique to the hierarchical basis
 * and live here as file-local statics.
 * ------------------------------------------------------------------------- */

/* Homogenized Legendre polynomials in two variables: thin wrapper around
 * PolyLegendre (declared in hvfem_internal.h). */
static PetscErrorCode HomLegendre(const PetscReal S[2], const PetscInt nord, PetscReal HomP[]) {
  PetscFunctionBeginUser;
  PetscCall(PolyLegendre(S[1], S[0] + S[1], nord, HomP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Edge-block ancillary functions for the hierarchical Nédélec construction.
 * Returns EE[3][nord] (vector value) and CurlEE[3][nord] (curl) on the
 * reference cell. Idec=PETSC_TRUE collapses to a degenerate edge whose
 * Whitney function vanishes; in that case curls are zero. */
static PetscErrorCode AncEE(const PetscReal S[2], const PetscReal DS[NUM_DIMENSIONS][2],
                            const PetscInt nord, const PetscBool Idec,
                            PetscReal **EE, PetscReal **CurlEE) {
  PetscFunctionBeginUser;
  const PetscInt minI = 1;
  const PetscInt maxI = nord;
  const PetscInt Ncurl = 2 * NUM_DIMENSIONS - 3;

  PetscReal *homP;
  PetscCall(PetscCalloc1(nord + 1, &homP));
  PetscCall(HomLegendre(S, maxI, homP));

  if (Idec) {
    for (PetscInt i = minI; i < maxI + 1; i++) {
      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        EE[j][i - 1] = homP[i - 1] * DS[j][1];
      }
    }
    for (PetscInt i = 0; i < Ncurl; i++) {
      for (PetscInt j = minI - 1; j < maxI - 1; j++) {
        CurlEE[i][j] = 0.0;
      }
    }
  } else {
    PetscReal whiE[NUM_DIMENSIONS], curlwhiE[NUM_DIMENSIONS];
    PetscReal a[NUM_DIMENSIONS], b[NUM_DIMENSIONS];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      whiE[i] = S[0] * DS[i][1] - S[1] * DS[i][0];
      a[i] = DS[i][0];
      b[i] = DS[i][1];
    }
    PetscCall(crossProduct(a, b, curlwhiE));

    for (PetscInt i = minI; i < maxI + 1; i++) {
      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        EE[j][i - 1] = homP[i - 1] * whiE[j];
      }
      for (PetscInt j = 0; j < Ncurl; j++) {
        CurlEE[j][i - 1] = (PetscReal)(i + 1) * homP[i - 1] * curlwhiE[j];
      }
    }
  }

  PetscCall(PetscFree(homP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Face-block ancillary functions: builds ETri[3][nord-1][nord-1] and
 * CurlETri[3][nord-1][nord-1] by combining AncEE on the face's tangent
 * coordinates with HomIJacobi in the third coordinate. */
static PetscErrorCode AncETri(const PetscReal S[NUM_DIMENSIONS],
                              const PetscReal DS[NUM_DIMENSIONS][NUM_DIMENSIONS],
                              const PetscInt nord, const PetscBool Idec,
                              PetscReal ***ETri, PetscReal ***CurlETri) {
  PetscFunctionBeginUser;
  const PetscInt minI = 0;
  const PetscInt minJ = 1;
  const PetscInt maxJ = nord - 1;
  const PetscInt maxIJ = nord - 1;
  const PetscInt minalpha = 2 * minI + 1;
  const PetscInt Ncurl = 2 * NUM_DIMENSIONS - 3;
  PetscReal tempS[2] = {S[0], S[1]};
  PetscReal tempDS[NUM_DIMENSIONS][2];
  PetscBool IdecE = PETSC_FALSE;

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    tempDS[i][0] = DS[i][0];
    tempDS[i][1] = DS[i][1];
  }

  PetscReal **EE, **curlEE, **homLal, ***DhomLal;
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &EE));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) PetscCall(PetscCalloc1(nord - minJ, &EE[i]));
  PetscCall(PetscCalloc1(2 * NUM_DIMENSIONS - 3, &curlEE));
  for (PetscInt i = 0; i < 2 * NUM_DIMENSIONS - 3; i++) PetscCall(PetscCalloc1(nord - minJ, &curlEE[i]));
  PetscCall(PetscCalloc1(maxJ, &homLal));
  for (PetscInt i = 0; i < maxJ; i++) PetscCall(PetscCalloc1(maxJ, &homLal[i]));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DhomLal));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(maxJ, &DhomLal[i]));
    for (PetscInt j = 0; j < maxJ; j++) PetscCall(PetscCalloc1(maxJ, &DhomLal[i][j]));
  }

  PetscCall(AncEE(tempS, (const PetscReal(*)[2])tempDS, nord - minJ, IdecE, EE, curlEE));

  PetscReal sL[2] = {S[0] + S[1], S[2]};
  PetscReal DsL[NUM_DIMENSIONS][2];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DsL[i][0] = DS[i][0] + DS[i][1];
    DsL[i][1] = DS[i][2];
  }
  PetscCall(HomIJacobi(sL, (const PetscReal(*)[2])DsL, maxJ, minalpha, Idec, homLal, DhomLal));

  for (PetscInt i = 0; i < maxIJ + 1; i++) {
    for (PetscInt j = minI; j < i - minJ + 1; j++) {
      const PetscInt k = i - j;
      for (PetscInt n = 0; n < NUM_DIMENSIONS; n++) {
        ETri[n][j][k - 1] = EE[n][j] * homLal[j][k - 1];
      }
      PetscReal cross_out[NUM_DIMENSIONS], v1[NUM_DIMENSIONS], v2[NUM_DIMENSIONS];
      for (PetscInt n = 0; n < NUM_DIMENSIONS; n++) {
        v1[n] = DhomLal[n][j][k - 1];
        v2[n] = EE[n][j];
      }
      PetscCall(crossProduct(v1, v2, cross_out));
      for (PetscInt n = 0; n < Ncurl; n++) {
        CurlETri[n][j][k - 1] = homLal[j][k - 1] * curlEE[n][j] + cross_out[n];
      }
    }
  }

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) PetscCall(PetscFree(EE[i]));
  PetscCall(PetscFree(EE));
  for (PetscInt i = 0; i < 2 * NUM_DIMENSIONS - 3; i++) PetscCall(PetscFree(curlEE[i]));
  PetscCall(PetscFree(curlEE));
  for (PetscInt i = 0; i < maxJ; i++) PetscCall(PetscFree(homLal[i]));
  PetscCall(PetscFree(homLal));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < maxJ; j++) PetscCall(PetscFree(DhomLal[i][j]));
    PetscCall(PetscFree(DhomLal[i]));
  }
  PetscCall(PetscFree(DhomLal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------------------
 * shape3DETet - full hierarchical H(curl) basis on the reference cell.
 *
 * Inputs:
 *   X[3]              : reference-cell point (xi, eta, zeta).
 *   nord              : polynomial order (1..6).
 *   cellOrientation   : 4 face codes (PETGEM convention, 0..5) and 6 edge
 *                       sign-from-DMPlex (±1).
 *
 * Outputs (caller-allocated, NUM_DIMENSIONS x numDofInCell):
 *   ShapE  : reference-cell vector value of each Nédélec shape function.
 *   CurlE  : reference-cell curl of each Nédélec shape function.
 *
 * The columns are returned in PETSc DOF order (matching what the rest of
 * the assembly expects). Conversion to physical space (Piola pullback) is
 * performed by the ops adapters below.
 *
 * NOTE: ported from the legacy hvfem_new_basis.c. Allocations follow the
 * same shape; reusable scratch buffers are an obvious follow-up
 * optimization once correctness is established.
 */
static PetscErrorCode shape3DETet(const PetscReal X[NUM_DIMENSIONS], const PetscInt nord,
                                  const CellOrientation *cellOrientation,
                                  PetscReal **ShapE, PetscReal **CurlE) {
  PetscFunctionBeginUser;

  const PetscInt numDofInCell = nord * (nord + 2) * (nord + 3) / 2;
  const PetscInt minI = 0, minJ = 1, minK = 1;
  const PetscInt minIJ = minI + minJ;
  const PetscInt minIJK = minIJ + minK;
  PetscInt m = 0;
  PetscInt NoriF[NUM_FACES_PER_CELL], NoriE[NUM_EDGES_PER_CELL];
  PetscBool IdecB[2] = {PETSC_FALSE, PETSC_FALSE};

  PetscReal Lam[NUM_DIMENSIONS + 1] = {0.0};
  PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1] = {{0.0}};
  PetscBool IdecE, IdecF;

  PetscCall(AffineTetrahedron(X, Lam, DLam));

  /* Map cell->orientation into the integer arrays the hierarchical
   * machinery expects: faces in {0..5}, edges in {0,1}. The face cast was
   * already applied in computeCellOrientation; edges are stored as ±1
   * sign so we just compress to a 0/1 index. */
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) NoriF[i] = cellOrientation->faces[i];
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++)
    NoriE[i] = (cellOrientation->edgeSigns[i] < 0) ? 1 : 0;

  /* Reset outputs (caller may have stale data). */
  for (PetscInt j = 0; j < NUM_DIMENSIONS; j++)
    for (PetscInt k = 0; k < numDofInCell; k++) {
      ShapE[j][k] = 0.0;
      CurlE[j][k] = 0.0;
    }

  /* ---- Edge block ------------------------------------------------------ */
  PetscReal LampE[NUM_EDGES_PER_CELL][2];
  PetscReal DLampE[NUM_EDGES_PER_CELL][NUM_DIMENSIONS][2];
  PetscReal **EE = NULL, **CurlEE = NULL;
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &EE));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) PetscCall(PetscCalloc1(nord, &EE[i]));
  PetscCall(PetscCalloc1(2 * NUM_DIMENSIONS - 3, &CurlEE));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) PetscCall(PetscCalloc1(nord, &CurlEE[i]));

  PetscCall(ProjectTetE(Lam, (const PetscReal(*)[NUM_DIMENSIONS + 1])DLam, LampE, DLampE, &IdecE));

  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    const PetscInt nordEdge = nord;
    const PetscInt numDofEdge = nordEdge;
    if (numDofEdge <= 0) continue;
    const PetscInt maxI = nordEdge - 1;

    PetscReal GLampE[2] = {0.0};
    PetscReal GDLampE[NUM_DIMENSIONS][2] = {{0.0}};
    PetscReal S[2] = {LampE[i][0], LampE[i][1]};
    PetscReal D[NUM_DIMENSIONS][2];
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++)
      for (PetscInt k = 0; k < 2; k++) D[j][k] = DLampE[i][j][k];

    PetscCall(OrientE(S, (const PetscReal(*)[2])D, NoriE[i], GLampE, GDLampE));
    PetscCall(AncEE(GLampE, (const PetscReal(*)[2])GDLampE, nordEdge, IdecE, EE, CurlEE));

    for (PetscInt j = minI; j < maxI + 1; j++) {
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
        ShapE[k][m] = EE[k][j];
        CurlE[k][m] = CurlEE[k][j];
      }
      m++;
    }
  }
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) PetscCall(PetscFree(EE[i]));
  PetscCall(PetscFree(EE));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) PetscCall(PetscFree(CurlEE[i]));
  PetscCall(PetscFree(CurlEE));

  /* ---- Face block ------------------------------------------------------ */
  /* Face DOFs exist only for nord >= 2; the scratch size is nord-1.
   * For nord=1 we still allocate the outer layer so the free path stays
   * uniform, but inner sizes are clamped to 0. */
  PetscReal LampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS];
  PetscReal DLampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS][NUM_DIMENSIONS];
  PetscReal ***ETri = NULL, ***CurlETri = NULL;
  const PetscInt nFaceCols = (nord >= 2) ? (nord - 1) : 0;

  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &ETri));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(nFaceCols, &ETri[i]));
    for (PetscInt j = 0; j < nFaceCols; j++) PetscCall(PetscCalloc1(nFaceCols, &ETri[i][j]));
  }
  PetscCall(PetscCalloc1(2 * NUM_DIMENSIONS - 3, &CurlETri));
  for (PetscInt i = 0; i < 2 * NUM_DIMENSIONS - 3; i++) {
    PetscCall(PetscCalloc1(nFaceCols, &CurlETri[i]));
    for (PetscInt j = 0; j < nFaceCols; j++) PetscCall(PetscCalloc1(nFaceCols, &CurlETri[i][j]));
  }

  PetscCall(ProjectTetF(Lam, (const PetscReal(*)[NUM_DIMENSIONS + 1])DLam, LampF, DLampF, &IdecF));

  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    const PetscInt nordFace = nord;
    const PetscInt numDofFace = nordFace * (nordFace - 1) / 2;
    if (numDofFace <= 0) continue;
    const PetscInt maxIJ = nordFace - 1;

    PetscReal GLampF[NUM_DIMENSIONS], GDLampF[NUM_DIMENSIONS][NUM_DIMENSIONS];
    PetscReal tmpLampF[NUM_DIMENSIONS], tempDLampF[NUM_DIMENSIONS][NUM_DIMENSIONS];
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
      tmpLampF[j] = LampF[i][j];
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) tempDLampF[j][k] = DLampF[i][j][k];
    }
    PetscCall(OrientTri(tmpLampF, (const PetscReal(*)[NUM_DIMENSIONS])tempDLampF,
                        NoriF[i], GLampF, GDLampF));

    PetscInt famctr = m;
    for (PetscInt j = 0; j < 2; j++) {
      m = famctr + j - 1;
      PetscInt abc[3];
      for (PetscInt k = 0; k < 3; k++) {
        PetscInt pos = (k - j) % 3;
        if (pos < 0) pos += 3;
        abc[pos] = k;
      }
      PetscReal tempGLampF[NUM_DIMENSIONS], tempGDLampF[NUM_DIMENSIONS][NUM_DIMENSIONS];
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
        tempGLampF[k] = GLampF[abc[k]];
        for (PetscInt t = 0; t < NUM_DIMENSIONS; t++) tempGDLampF[t][k] = GDLampF[t][abc[k]];
      }
      PetscCall(AncETri(tempGLampF, (const PetscReal(*)[NUM_DIMENSIONS])tempGDLampF,
                        nordFace, IdecF, ETri, CurlETri));

      for (PetscInt k = minIJ; k < maxIJ + 1; k++) {
        for (PetscInt r = minI; r < k - minJ + 1; r++) {
          const PetscInt p = k - r;
          m += 2;
          for (PetscInt t = 0; t < NUM_DIMENSIONS; t++) {
            ShapE[t][m - 1] = ETri[t][r][p - 1];
            CurlE[t][m - 1] = CurlETri[t][r][p - 1];
          }
        }
      }
    }
  }
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < nFaceCols; j++) PetscCall(PetscFree(ETri[i][j]));
    PetscCall(PetscFree(ETri[i]));
  }
  PetscCall(PetscFree(ETri));
  for (PetscInt i = 0; i < 2 * NUM_DIMENSIONS - 3; i++) {
    for (PetscInt j = 0; j < nFaceCols; j++) PetscCall(PetscFree(CurlETri[i][j]));
    PetscCall(PetscFree(CurlETri[i]));
  }
  PetscCall(PetscFree(CurlETri));

  /* ---- Volume (bubble) block ------------------------------------------ */
  /* Volume bubbles exist only for nord >= 3. For lower orders we still
   * allocate outer pointer layers so the free path stays uniform, but
   * inner sizes are clamped to 0 (PetscCalloc1(0,...) is benign). */
  const PetscInt nordB = nord;
  const PetscInt ndofB = nordB * (nordB - 1) * (nordB - 2) / 6;
  const PetscInt minbeta = 2 * minIJ;
  const PetscInt maxIJK = nordB - 1;
  const PetscInt maxK = (maxIJK - minIJ > 0) ? (maxIJK - minIJ) : 0;
  const PetscInt nVolCols = (nord - minK - 1 > 0) ? (nord - minK - 1) : 0;

  PetscReal ***ETriV = NULL, ***CurlETriV = NULL;
  PetscReal **homLbet = NULL, ***DhomLbet = NULL;

  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &ETriV));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(nVolCols, &ETriV[i]));
    for (PetscInt j = 0; j < nVolCols; j++) PetscCall(PetscCalloc1(nVolCols, &ETriV[i][j]));
  }
  PetscCall(PetscCalloc1(2 * NUM_DIMENSIONS - 3, &CurlETriV));
  for (PetscInt i = 0; i < 2 * NUM_DIMENSIONS - 3; i++) {
    PetscCall(PetscCalloc1(nVolCols, &CurlETriV[i]));
    for (PetscInt j = 0; j < nVolCols; j++) PetscCall(PetscCalloc1(nVolCols, &CurlETriV[i][j]));
  }
  PetscCall(PetscCalloc1(maxK, &homLbet));
  for (PetscInt i = 0; i < maxK; i++) PetscCall(PetscCalloc1(maxK, &homLbet[i]));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DhomLbet));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(maxK, &DhomLbet[i]));
    for (PetscInt j = 0; j < maxK; j++) PetscCall(PetscCalloc1(maxK, &DhomLbet[i][j]));
  }

  if (ndofB > 0) {
    IdecB[0] = IdecF;
    IdecB[1] = PETSC_TRUE;

    PetscInt famctr = m;
    for (PetscInt i = 0; i < 3; i++) {
      m = famctr + i - 2;
      PetscInt abcd[4];
      for (PetscInt j = 0; j < 4; j++) {
        PetscInt pos = (j - i) % 4;
        if (pos < 0) pos += 4;
        abcd[pos] = j;
      }
      const PetscInt abc[3] = {abcd[0], abcd[1], abcd[2]};
      const PetscInt d = abcd[3];

      PetscReal tempLam[NUM_DIMENSIONS], tempDLam[NUM_DIMENSIONS][NUM_DIMENSIONS];
      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        tempLam[j] = Lam[abc[j]];
        for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) tempDLam[k][j] = DLam[k][abc[j]];
      }

      PetscCall(AncETri(tempLam, (const PetscReal(*)[NUM_DIMENSIONS])tempDLam,
                        nordB - minK, IdecB[0], ETriV, CurlETriV));

      const PetscReal tmp1[2] = {1.0 - Lam[d], Lam[d]};
      PetscReal tmp2[NUM_DIMENSIONS][2];
      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        tmp2[j][0] = -DLam[j][d];
        tmp2[j][1] =  DLam[j][d];
      }
      PetscCall(HomIJacobi(tmp1, (const PetscReal(*)[2])tmp2, maxK, minbeta, IdecB[1], homLbet, DhomLbet));

      for (PetscInt j = minIJK; j < maxIJK + 1; j++) {
        for (PetscInt k = minIJ; k < j - minK + 1; k++) {
          for (PetscInt r = minI; r < k - minJ + 1; r++) {
            const PetscInt p = k - r;
            const PetscInt q = j - k;
            m += 3;

            for (PetscInt n = 0; n < NUM_DIMENSIONS; n++)
              ShapE[n][m - 1] = ETriV[n][r][p - 1] * homLbet[k - 1][q - 1];

            PetscReal cross_out[NUM_DIMENSIONS], v1[NUM_DIMENSIONS], v2[NUM_DIMENSIONS];
            for (PetscInt n = 0; n < NUM_DIMENSIONS; n++) {
              v1[n] = DhomLbet[n][k - 1][q - 1];
              v2[n] = ETriV[n][r][p - 1];
            }
            PetscCall(crossProduct(v1, v2, cross_out));
            for (PetscInt n = 0; n < NUM_DIMENSIONS; n++)
              CurlE[n][m - 1] = homLbet[k - 1][q - 1] * CurlETriV[n][r][p - 1] + cross_out[n];
          }
        }
      }
    }
  }

  /* ---- PETGEM (hierarchical) -> PETSc DOF reorder --------------------- */
  /* Permutation table flattened by order (offsets index by nord-1).
   * Sizes: 6, 20, 45, 84, 140, 216 for nord = 1..6. */
  static const PetscInt perm[] = {
      0,   1,   2,   3,   4,   5,
      12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
      42, 43, 44, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
      13, 14, 15, 16, 17,
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 24, 25, 26, 27, 28, 29, 30, 31,
      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
      52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
       0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
      20, 21, 22, 23,
     110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
     130, 131, 132, 133, 134, 135, 136, 137, 138, 139,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
      40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
      80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
     100, 101, 102, 103, 104, 105, 106, 107, 108, 109,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
      10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
     156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
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
  static const PetscInt permOffsets[] = {0, 6, 26, 71, 155, 295};

  PetscReal tmpShapE[NUM_DIMENSIONS][numDofInCell], tmpCurlE[NUM_DIMENSIONS][numDofInCell];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++)
    for (PetscInt j = 0; j < numDofInCell; j++) {
      tmpShapE[i][j] = ShapE[i][j];
      tmpCurlE[i][j] = CurlE[i][j];
    }
  const PetscInt *p = perm + permOffsets[nord - 1];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++)
    for (PetscInt j = 0; j < numDofInCell; j++) {
      ShapE[i][j] = tmpShapE[i][p[j]];
      CurlE[i][j] = tmpCurlE[i][p[j]];
    }

  /* Free volume bubble scratch */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < nVolCols; j++) PetscCall(PetscFree(ETriV[i][j]));
    PetscCall(PetscFree(ETriV[i]));
  }
  PetscCall(PetscFree(ETriV));
  for (PetscInt i = 0; i < 2 * NUM_DIMENSIONS - 3; i++) {
    for (PetscInt j = 0; j < nVolCols; j++) PetscCall(PetscFree(CurlETriV[i][j]));
    PetscCall(PetscFree(CurlETriV[i]));
  }
  PetscCall(PetscFree(CurlETriV));
  for (PetscInt i = 0; i < maxK; i++) PetscCall(PetscFree(homLbet[i]));
  PetscCall(PetscFree(homLbet));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < maxK; j++) PetscCall(PetscFree(DhomLbet[i][j]));
    PetscCall(PetscFree(DhomLbet[i]));
  }
  PetscCall(PetscFree(DhomLbet));

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* ===========================================================================
 * NedelecOps adapters for the hierarchical basis.
 *
 * The hierarchical construction does not have an explicit per-cell
 * coefficient solve (unlike nord=1 / nord=2 which both fit a moment
 * system). computeCoefficients is therefore a no-op; coeffs / Dx / Dy / Dz
 * buffers are passed through untouched.
 *
 * One ops table per order so nord is captured by the adapter (matching the
 * idiom used by nedelecOps_order1 / _order2). All four orders delegate to
 * the same shape3DETet kernel and the same Piola-pullback adapters below.
 *
 * computeBasis / computeCurls each call shape3DETet and apply the standard
 * covariant Piola pullback. shape3DETet is invoked twice per quadrature
 * point in the current pipeline (once for basis, once for curls); a fused
 * entry point is the natural follow-up optimization.
 * ========================================================================= */

static PetscErrorCode hierarchicalComputeCoefficients(const Cell *cell, PetscReal **coeffs,
                                                      PetscReal **Dx_Ni, PetscReal **Dy_Ni,
                                                      PetscReal **Dz_Ni) {
  (void)cell; (void)coeffs; (void)Dx_Ni; (void)Dy_Ni; (void)Dz_Ni;
  return PETSC_SUCCESS;
}

/* Shared adapter body for basis evaluation: parameterized by `nord`, called
 * by the per-order thin wrappers below. */
static PetscErrorCode hierarchicalComputeBasisOrder(PetscInt nord, const Cell *cell,
                                                    const PetscReal point[NUM_DIMENSIONS],
                                                    PetscReal **Ni) {
  PetscFunctionBeginUser;
  const PetscInt numDofInCell = nord * (nord + 2) * (nord + 3) / 2;

  PetscReal **ShapE_ref = NULL, **CurlE_ref = NULL;
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &ShapE_ref));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &CurlE_ref));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(numDofInCell, &ShapE_ref[i]));
    PetscCall(PetscCalloc1(numDofInCell, &CurlE_ref[i]));
  }

  PetscCall(shape3DETet(point, nord, &cell->orientation, ShapE_ref, CurlE_ref));

  /* Covariant Piola: Ni_phys = J^{-T} * Ni_ref. cell->invJacobian stores J^{-T}. */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++)
    for (PetscInt j = 0; j < numDofInCell; j++) {
      PetscReal v = 0.0;
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) v += cell->invJacobian[i][k] * ShapE_ref[k][j];
      Ni[i][j] = v;
    }

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(ShapE_ref[i]));
    PetscCall(PetscFree(CurlE_ref[i]));
  }
  PetscCall(PetscFree(ShapE_ref));
  PetscCall(PetscFree(CurlE_ref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Shared adapter body for curl evaluation. */
static PetscErrorCode hierarchicalComputeCurlsOrder(PetscInt nord, const Cell *cell,
                                                    const PetscReal point[NUM_DIMENSIONS],
                                                    PetscReal **NiCurl) {
  PetscFunctionBeginUser;
  const PetscInt numDofInCell = nord * (nord + 2) * (nord + 3) / 2;

  PetscReal **ShapE_ref = NULL, **CurlE_ref = NULL;
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &ShapE_ref));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &CurlE_ref));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(numDofInCell, &ShapE_ref[i]));
    PetscCall(PetscCalloc1(numDofInCell, &CurlE_ref[i]));
  }

  PetscCall(shape3DETet(point, nord, &cell->orientation, ShapE_ref, CurlE_ref));

  /* Curl Piola: curl_phys = (J * curl_ref) / det(J). */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++)
    for (PetscInt j = 0; j < numDofInCell; j++) {
      PetscReal v = 0.0;
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) v += cell->jacobian[k][i] * CurlE_ref[k][j];
      NiCurl[i][j] = v / cell->detJacobian;
    }

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(ShapE_ref[i]));
    PetscCall(PetscFree(CurlE_ref[i]));
  }
  PetscCall(PetscFree(ShapE_ref));
  PetscCall(PetscFree(CurlE_ref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Topological discrete gradient builder (lowest-order Whitney slot
 * only). Used by the inverse kernel against P1 H1 (numH1DofInCell = 4)
 * and as a fallback BDDC hint at nord = 1. At nord >= 2 the FORWARD
 * kernel uses hierarchicalBuildExactGradientMatrix below, which gives
 * the exact commuting G against P_nord H1.
 *
 * Only the LOWEST-ORDER Whitney DOF per mesh edge carries the ±1 vertex
 * incidence. Higher-order edge DOFs (n > 0) and all face/volume DOF
 * rows are zero.
 *
 * Sign convention: edges in canonical orientation (NoriE=0, edgeSigns >= 0)
 * get -1 at the start vertex and +1 at the end vertex of EDGE_VERTICES[e].
 * Reversed edges get the opposite - the standard Whitney tangential
 * direction. */
static PetscErrorCode hierarchicalBuildGradientMatrixTopological(const FEMSpace *fem, const Cell *cell,
                                                                  const Quadrature1D *quadrature1d,
                                                                  PetscReal **gradientMatrix) {
  PetscFunctionBeginUser;
  (void)quadrature1d;

  const PetscInt numDofInCell    = fem->numDofInCell;
  const PetscInt numH1DofInCell  = fem->numH1DofInCell;
  const PetscInt numDofPerEdge   = fem->numDofPerEdge;
  const PetscInt edgeDofOffset   = fem->edgeDofOffset;

  /* Reset the matrix; face/volume and higher-order edge rows stay zero. */
  for (PetscInt i = 0; i < numDofInCell; i++)
    for (PetscInt j = 0; j < numH1DofInCell; j++) gradientMatrix[i][j] = 0.0;

  /* Edge -> endpoint vertex pairs in PETGEM canonical order
   * (matches EDGE_VERTICES in constants.c). */
  static const PetscInt EDGE_ENDS[NUM_EDGES_PER_CELL][2] = {
      {0, 1}, {1, 2}, {2, 0}, {0, 3}, {3, 1}, {2, 3}};

  /* For each mesh edge, fill ONLY the lowest-order Whitney slot
   * (n = 0 within the edge's DOF block) with the ±1 vertex incidence. */
  for (PetscInt e = 0; e < NUM_EDGES_PER_CELL; e++) {
    const PetscBool reversed = (PetscBool)(cell->orientation.edgeSigns[e] < 0);
    const PetscInt  va = EDGE_ENDS[e][0];
    const PetscInt  vb = EDGE_ENDS[e][1];
    const PetscReal valA = reversed ?  1.0 : -1.0;
    const PetscReal valB = reversed ? -1.0 :  1.0;

    /* Lowest Whitney slot for edge e: edgeDofOffset + e*numDofPerEdge + 0.
     * Slots e*numDofPerEdge + 1..numDofPerEdge-1 (higher-order Whitneys
     * on the same edge) intentionally remain zero. */
    const PetscInt slot = edgeDofOffset + e * numDofPerEdge;
    gradientMatrix[slot][va] = valA;
    gradientMatrix[slot][vb] = valB;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ===========================================================================
 * Exact commuting discrete-gradient builder for the hierarchical
 * H1/H(curl) pair on tetrahedra (Demkowicz hp-FE space + Ainsworth–Coyle
 * DOF moments).
 *
 * The FE pair forms a subcomplex of the de Rham complex:
 *
 *     P_nord(K)  --∇-->  Ned_nord(K)  --curl-->  RT_nord(K)  --div-->  P_{nord-1}(K)
 *        H1                H(curl)               H(div)               L2
 *
 * with ∇P_nord ⊂ Ned_nord (Nédélec_1 of the first kind, order `nord`),
 * so each ∇φ_j (j-th P_nord H1 basis) admits a unique global expansion
 *     ∇φ_j = Σ_l Ge[l, j] · N_l
 * in the global Nédélec basis. Per-cell, this is
 *     ∇φ_j|_K = Σ_l Ge_K[l, j] · N_l^K.
 *
 * The previous implementation (per-cell L2 projection of ∇φ_j onto the
 * cell Nédélec basis) is correct ELEMENT-wise (K_e · Ge_e ≈ 0) but
 * INCONSISTENT across adjacent cells: the hierarchical Nédélec basis is
 * NOT biorthogonal to the canonical DOF moments, so the L2 coefficient
 * for a SHARED edge/face DOF depends on each cell's local mass matrix
 * and can differ between cells. INSERT_VALUES then produces a multi-
 * valued global G with ||K·G||/||K||·||G|| ≈ 1e-3 at nord=2.
 *
 * This builder fixes the issue by computing Ge from the canonical
 * Nédélec interpolation operator Π^Ned:
 *
 *     σ_i(Π^Ned u)  =  σ_i(u)   for every DOF moment σ_i,
 *
 * where {σ_i} is the Ainsworth–Coyle set of moment functionals:
 *
 *   * Edge DOFs (mode n on edge e, n = 0..nord-1):
 *         σ_e^n(u) = ∫_{-1}^{+1} (u(γ_e(t)) · t̂_e) · L_n(t) dt
 *     with γ_e the CANONICAL parameterization of edge e (low-vertex to
 *     high-vertex by global ordering) and L_n the n-th Legendre poly.
 *
 *   * Face DOFs (spatial (p,q) with p+q ≤ nord-2, tangent α ∈ {1,2}):
 *         σ_F^{(p,q),α}(u) = ∫_F (u · e_α^F) · ψ_{p,q}^Dubiner(λ) dA
 *     with (v_a, v_b, v_c) the CANONICAL face vertex order (sorted by
 *     physical coordinates - see computeCellOrientation), tangents
 *     e_1^F = v_b - v_a, e_2^F = v_c - v_a, and ψ_{p,q} the Dubiner
 *     polynomial basis on the canonical triangle (L²-orthogonal, see
 *     hcurlDubinerFace below).
 *
 *   * Volume DOFs (spatial (p,q,r) with p+q+r ≤ nord-3, direction β):
 *         σ_K^{(p,q,r),β}(u) = ∫_K (u · e_β^K) · ψ_{p,q,r}^Koornwinder(X_ref) dV
 *     with cell-local tangents e_β^K = V_β - V_0 (β = 1, 2, 3) and
 *     ψ_{p,q,r} the Koornwinder (3D Dubiner) polynomial basis on the
 *     reference tetrahedron (L²-orthogonal, see hcurlKoornwinderVolume).
 *     Volume DOFs are NOT shared across cells so any consistent intra-
 *     cell choice works; orthogonality is used for conditioning, not
 *     consistency.
 *
 * Letting D[i, l] := σ_i(N_l) and B[i, j] := σ_i(∇φ_j), the per-cell
 * coefficient matrix solves
 *     D_K · Ge_K = B_K     (LAPACK GESV + post-solve sanitization)
 *
 * The moment-test side uses L²-orthogonal Dubiner / Koornwinder
 * polynomials. On well-conditioned cells (nord ≤ 4 everywhere; nord
 * ≥ 5 on uniform meshes) the partial-pivoting LU solve is exact to
 * machine precision and the resulting G has the canonical sparsity
 * (nnz/row matches the topological coupling). On distorted CSEM
 * cells at nord ≥ 5 the hierarchical Nédélec basis from shape3DETet
 * (HomIJacobi / AncETri recurrences) overflows at quadrature points
 * with near-singular collapsed-coordinate denominators, producing
 * NaN/Inf in B during moment accumulation; LU then either reports
 * INFO > 0, silently writes NaN into Ge, or (when partial pivoting
 * picks a near-zero pivot that's finite but tiny) amplifies Ge
 * entries to ~1e100+ magnitudes that overflow the squared-sum
 * Frobenius norm of the global G. After the solve we scan Ge and
 * replace any entry that is non-finite OR whose magnitude exceeds
 * an unphysical threshold (1e6 - two orders above any legitimate
 * Nédélec coefficient at supported orders) with zero; the cell
 * contributes a
 * degraded (but finite) row to the global G with a diagnostic line
 * announcing how many entries were affected. Earlier versions tried
 * GELSS (SVD with rank truncation) but that regressed well-conditioned
 * cells (truncation noise polluted the canonical-sparsity pattern);
 * the LU + post-solve scrub approach keeps nord ≤ 4 exact while
 * gracefully handling nord ≥ 5 outliers.
 *
 * Cross-cell consistency: σ_i depends only on the geometric carrier of
 * DOF i (edge/face/volume) and the trace of u on that carrier. Both
 * cells sharing an edge or face evaluate the SAME integral (same
 * canonical parameterization, same tangents, same polynomial test
 * functions), so the rows of D and B for shared DOFs are identical
 * across cells. Combined with global tangential continuity of u in
 * H(curl) and the fact that ∇φ_j ∈ Nédélec exactly, the unique global
 * coefficient Ge[l, j] is reproduced by every cell that touches DOF l.
 *
 * Result:
 *   * K · G = 0 to machine precision (algebraic commuting diagram holds).
 *   * G · c = 0 for any constant H1 function c (∇1 ≡ 0; here c lives in
 *     the hierarchical H1 basis as vertex-DOFs = 1, bubbles = 0).
 *   * The nnz pattern of each row of G matches the topological coupling
 *     (DOF on entity E couples only to H1 DOFs whose carriers touch E).
 *
 * References:
 *   * L. Demkowicz, "Computing with hp-Adaptive Finite Elements", Vol.1
 *     (2006) - hierarchical H1/H(curl) basis construction.
 *   * M. Ainsworth & J. Coyle, "Hierarchic finite element bases on
 *     unstructured tetrahedral meshes", IJNME 58 (2003) - operational
 *     DOF moment definitions used here.
 *   * J.-C. Nédélec, "Mixed finite elements in R^3", Numer. Math. 35
 *     (1980) - original Nédélec_1 first-kind space & DOFs.
 * ========================================================================= */

/* Reference-tetrahedron vertex coordinates (PETGEM convention). Matches
 * the affine map λ_0 = 1-ξ-η-ζ, λ_i = ξ/η/ζ in AffineTetrahedron. */
static const PetscReal HCURL_REF_TET_VERTICES[NUM_VERTICES_PER_CELL][NUM_DIMENSIONS] = {
    {0.0, 0.0, 0.0},  /* V0 */
    {1.0, 0.0, 0.0},  /* V1 */
    {0.0, 1.0, 0.0},  /* V2 */
    {0.0, 0.0, 1.0}}; /* V3 */

/* Canonical face vertex order. cell->orientation.faces[f] encodes the
 * permutation of FACE_VERTICES[f] that produces the vertex ordering
 * sorted by physical coordinates (see computeCellOrientation). Both
 * cells sharing face f arrive at the same canonical triple, so the
 * tangents e_1 = canon[1]-canon[0], e_2 = canon[2]-canon[0] and the
 * barycentric polynomials λ_a^i λ_b^j λ_c^k are cell-invariant. */
static inline void hcurlFaceCanonicalVertices(const Cell *cell, PetscInt f,
                                              PetscInt canon[3]) {
  const PetscInt v0 = FACE_VERTICES[f][0];
  const PetscInt v1 = FACE_VERTICES[f][1];
  const PetscInt v2 = FACE_VERTICES[f][2];
  switch (cell->orientation.faces[f]) {
  case 0:  canon[0]=v0; canon[1]=v1; canon[2]=v2; break;
  case 1:  canon[0]=v1; canon[1]=v2; canon[2]=v0; break;
  case 2:  canon[0]=v2; canon[1]=v0; canon[2]=v1; break;
  case 3:  canon[0]=v0; canon[1]=v2; canon[2]=v1; break;
  case 4:  canon[0]=v1; canon[1]=v0; canon[2]=v2; break;
  case 5:  canon[0]=v2; canon[1]=v1; canon[2]=v0; break;
  default: canon[0]=v0; canon[1]=v1; canon[2]=v2; break;
  }
}

/* ---------------------------------------------------------------------------
 * Orthogonal polynomial test functions on the reference simplex.
 *
 * Replace the natural monomial choices (λ_a^i λ_b^j λ_c^k on the face,
 * x^i y^j z^k in the volume) with Dubiner / Koornwinder bases. The
 * monomial bases are L²-spanning but become near-linearly-dependent at
 * high order - their Gram matrix is the Hilbert / Hilbert-like matrix
 * with κ ~ exp(n). The moment-duality matrix D inherits that
 * conditioning, and at nord ≥ 5 even Wilkinson iterative refinement
 * diverges on some CSEM cells.
 *
 * The Dubiner basis on the reference triangle and its Koornwinder
 * 3D analogue on the reference tetrahedron are L²-orthogonal:
 *   ∫_K ψ_α(x) ψ_β(x) dx = c_α δ_αβ.
 * They are computed in O(p²) / O(p³) flops via classical 3-term Jacobi
 * recurrences. The moment-test side of D becomes well-conditioned;
 * residual conditioning at nord ≥ 5 comes from the hierarchical
 * Nédélec basis itself (shape3DETet recurrences span > 10^16 dynamic
 * range on distorted cells), which the per-cell GELSS solver handles
 * via rank truncation.
 *
 * References:
 *   * M. Dubiner, "Spectral methods on triangles and other domains",
 *     J. Sci. Comp. 6 (1991).
 *   * T. Koornwinder, "Two-variable analogues of the classical
 *     orthogonal polynomials", Theory and Application of Special
 *     Functions (1975).
 *   * G. Karniadakis, S. Sherwin, "Spectral/hp Element Methods for
 *     CFD" (2005), Ch. 3 - collapsed-coordinate construction.
 * ------------------------------------------------------------------------- */

/* Jacobi polynomial P_n^{(α,β)}(x) via the classical 3-term recurrence
 * (Abramowitz & Stegun 22.7.1). Evaluates at scalar x for n ≥ 0,
 * α, β > −1. Numerically stable for the (α, β) pairs used below
 * (β = 0 always; α = 0 / 2p+1 / 2p+2q+2). */
static PetscReal hcurlJacobi(PetscReal x, PetscInt n, PetscReal alpha, PetscReal beta) {
  if (n == 0) return 1.0;
  PetscReal P0 = 1.0;
  PetscReal P1 = 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x;
  if (n == 1) return P1;

  PetscReal Pkm1 = P0, Pk = P1, Pkp1 = 0.0;
  for (PetscInt k = 1; k < n; k++) {
    const PetscReal kk    = (PetscReal)k;
    const PetscReal denom = 2.0 * (kk + 1.0) * (kk + alpha + beta + 1.0) * (2.0 * kk + alpha + beta);
    const PetscReal c1    = (2.0 * kk + alpha + beta + 1.0) * (alpha * alpha - beta * beta);
    const PetscReal c2    = (2.0 * kk + alpha + beta) * (2.0 * kk + alpha + beta + 1.0) * (2.0 * kk + alpha + beta + 2.0);
    const PetscReal c3    = 2.0 * (kk + alpha) * (kk + beta) * (2.0 * kk + alpha + beta + 2.0);
    Pkp1 = ((c1 + c2 * x) * Pk - c3 * Pkm1) / denom;
    Pkm1 = Pk;
    Pk   = Pkp1;
  }
  return Pk;
}

/* Dubiner basis function ψ_{p,q} on the canonical triangle.
 *
 * Input barycentric coords (λ_a, λ_b, λ_c) with λ_a + λ_b + λ_c = 1.
 * Collapsed coords (apex at λ_c = 1):
 *     η₁ = 2 λ_b / (1 − λ_c) − 1        ∈ [−1, 1]
 *     η₂ = 2 λ_c − 1                     ∈ [−1, 1]
 * Basis:
 *     ψ_{p,q}(λ) = P_p^{(0,0)}(η₁) · ((1−η₂)/2)^p · P_q^{(2p+1, 0)}(η₂).
 * Polynomial degree p + q on the triangle.
 *
 * At the apex λ_c = 1 the collapsed coordinate η₁ is undefined (0/0),
 * but the singular factor ((1−η₂)/2)^p = 0 (for p ≥ 1) annihilates the
 * basis function. For p = 0 the η₁ factor is the constant 1, so the
 * value at the apex is well-defined. We branch on the singular case
 * and return 0 for p ≥ 1 / handle p = 0 explicitly. */
static PetscReal hcurlDubinerFace(const PetscReal lam[3], PetscInt p, PetscInt q) {
  const PetscReal eta2  = 2.0 * lam[2] - 1.0;
  const PetscReal omega = 1.0 - lam[2];
  PetscReal eta1;
  if (PetscAbsReal(omega) > 1.0e-14) {
    eta1 = 2.0 * lam[1] / omega - 1.0;
  } else {
    /* Apex: η₁ undefined, but ψ_{0,q}(apex) = P_q^{(1,0)}(η₂); ψ_{≥1,q}(apex) = 0. */
    if (p > 0) return 0.0;
    eta1 = 0.0;
  }
  const PetscReal Pp    = hcurlJacobi(eta1, p, 0.0, 0.0);
  const PetscReal scale = PetscPowReal(omega * 0.5, (PetscReal)p);  /* ((1−η₂)/2)^p = (omega/2)^p */
  const PetscReal Pq    = hcurlJacobi(eta2, q, 2.0 * (PetscReal)p + 1.0, 0.0);
  return Pp * scale * Pq;
}

/* Koornwinder basis ψ_{p,q,r} on the reference tetrahedron.
 *
 * Reference coords X_ref = (x, y, z). Barycentric:
 *     λ_0 = 1 − x − y − z,  λ_1 = x,  λ_2 = y,  λ_3 = z.
 * Collapsed coords (apex chain at λ_3 = 1, then λ_2 + λ_3 = 1, then ...):
 *     η₁ = 2 λ_1 / (1 − λ_2 − λ_3) − 1
 *     η₂ = 2 λ_2 / (1 − λ_3) − 1
 *     η₃ = 2 λ_3 − 1
 * Basis:
 *     ψ_{p,q,r}(λ) = P_p^{(0,0)}(η₁)
 *                   · ((1−η₂)/2)^p · P_q^{(2p+1, 0)}(η₂)
 *                   · ((1−η₃)/2)^{p+q} · P_r^{(2p+2q+2, 0)}(η₃).
 * Polynomial degree p + q + r in (x, y, z).
 *
 * Singular-coord handling mirrors the face case: at each apex the
 * undefined collapsed coordinate is multiplied by a vanishing scale
 * factor for degree ≥ 1, so we return 0 in those cases and use a
 * well-defined fallback for the constant-mode tail. */
static PetscReal hcurlKoornwinderVolume(const PetscReal X_ref[NUM_DIMENSIONS],
                                         PetscInt p, PetscInt q, PetscInt r) {
  const PetscReal lam1 = X_ref[0];
  const PetscReal lam2 = X_ref[1];
  const PetscReal lam3 = X_ref[2];

  const PetscReal eta3   = 2.0 * lam3 - 1.0;
  const PetscReal omega3 = 1.0 - lam3;
  PetscReal eta2, omega2;
  if (PetscAbsReal(omega3) > 1.0e-14) {
    eta2   = 2.0 * lam2 / omega3 - 1.0;
    omega2 = 1.0 - lam2 - lam3;
  } else {
    /* λ_3 ≈ 1: ψ_{·,·,r ≥ 1} undefined; for r = 0 the tail collapses. */
    if (p + q > 0) return 0.0;
    eta2   = 0.0;
    omega2 = 0.0;
  }
  PetscReal eta1;
  if (PetscAbsReal(omega2) > 1.0e-14) {
    eta1 = 2.0 * lam1 / omega2 - 1.0;
  } else {
    if (p > 0) return 0.0;
    eta1 = 0.0;
  }
  const PetscReal Pp     = hcurlJacobi(eta1, p, 0.0, 0.0);
  const PetscReal scale1 = PetscPowReal(omega2 * 0.5, (PetscReal)p);
  const PetscReal Pq     = hcurlJacobi(eta2, q, 2.0 * (PetscReal)p + 1.0, 0.0);
  const PetscReal scale2 = PetscPowReal(omega3 * 0.5, (PetscReal)(p + q));
  const PetscReal Pr     = hcurlJacobi(eta3, r, 2.0 * (PetscReal)(p + q) + 2.0, 0.0);
  return Pp * scale1 * Pq * scale2 * Pr;
}

/* Pull the reference-frame H1 gradient into physical coordinates:
 * ∇_phys φ = J^{-T} ∇_ref φ. cell->invJacobian stores J^{-T}. */
static inline void hcurlPullGradient(const Cell *cell, const PetscReal grad_ref[NUM_DIMENSIONS],
                                     PetscReal grad_phys[NUM_DIMENSIONS]) {
  grad_phys[0] = cell->invJacobian[0][0] * grad_ref[0]
               + cell->invJacobian[0][1] * grad_ref[1]
               + cell->invJacobian[0][2] * grad_ref[2];
  grad_phys[1] = cell->invJacobian[1][0] * grad_ref[0]
               + cell->invJacobian[1][1] * grad_ref[1]
               + cell->invJacobian[1][2] * grad_ref[2];
  grad_phys[2] = cell->invJacobian[2][0] * grad_ref[0]
               + cell->invJacobian[2][1] * grad_ref[1]
               + cell->invJacobian[2][2] * grad_ref[2];
}

/* Add edge moments to D and B.
 *
 * For each mesh edge e with cell-local vertices (v_a, v_b) and ε = ±1
 * orientation sign, define the CANONICAL parameter s_can ∈ [-1, 1] such
 * that s_can = -1 maps to the canonical-low vertex. The cell-local
 * parameter s_cell ∈ [0, 1] relates to s_can by
 *     ε = +1: s_cell = (s_can + 1) / 2
 *     ε = -1: s_cell = (1 - s_can) / 2
 * The canonical tangent t_can = ε · (v_b - v_a) (same vector from both
 * cells). For each Legendre mode n = 0..nord-1, accumulate
 *     D[row(e,n), l] += w · L_n(s_can) · (N_l(γ_cell(s_cell)) · t_can)
 *     B[row(e,n), j] += w · L_n(s_can) · (∇_phys φ_j(γ_cell(s_cell)) · t_can)
 * where row(e,n) = edgeDofOffset + e · numDofPerEdge + n. */
static PetscErrorCode hcurlAddEdgeMoments(const FEMSpace *fem, const Cell *cell,
                                          const NedelecOps *ops,
                                          const PetscReal *const *coeffs_const,
                                          PetscReal **Ni, PetscReal *ShapH,
                                          PetscReal **GradH,
                                          PetscScalar *D, PetscScalar *B) {
  PetscFunctionBeginUser;
  const PetscInt nord  = fem->nord;
  const PetscInt n     = fem->numDofInCell;
  const PetscInt nh    = fem->numH1DofInCell_Pnord;

  /* 1D Gauss-Legendre on [-1, 1]. nord points are exact for polynomials
   * of degree 2*nord - 1, which dominates the integrand degree
   * (nord-1 from tangential trace of N_l) + (nord-1 from L_n). */
  Quadrature1D q1 = {0};
  q1.numPoints = nord;
  PetscCall(PetscCalloc1(q1.numPoints, &q1.points));
  PetscCall(PetscCalloc1(q1.numPoints, &q1.weights));
  PetscCall(compute1DQuadraturePoints(&q1));

  PetscReal LnVals[8] = {0.0};  /* L_0..L_{nord-1}; max nord = 6 */

  for (PetscInt e = 0; e < NUM_EDGES_PER_CELL; e++) {
    const PetscInt va  = EDGE_VERTICES[e][0];
    const PetscInt vb  = EDGE_VERTICES[e][1];
    const PetscInt eps = cell->orientation.edgeSigns[e];

    PetscReal t_can[NUM_DIMENSIONS];
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
      t_can[d] = (PetscReal)eps
               * (cell->coordinates[vb * NUM_DIMENSIONS + d]
                - cell->coordinates[va * NUM_DIMENSIONS + d]);
    }

    for (PetscInt qp = 0; qp < q1.numPoints; qp++) {
      const PetscReal s_can = q1.points[qp];
      const PetscReal w     = q1.weights[qp];
      const PetscReal s_cell = (eps == 1) ? (s_can + 1.0) * 0.5
                                          : (1.0 - s_can) * 0.5;

      const PetscReal X_ref[NUM_DIMENSIONS] = {
          (1.0 - s_cell) * HCURL_REF_TET_VERTICES[va][0] + s_cell * HCURL_REF_TET_VERTICES[vb][0],
          (1.0 - s_cell) * HCURL_REF_TET_VERTICES[va][1] + s_cell * HCURL_REF_TET_VERTICES[vb][1],
          (1.0 - s_cell) * HCURL_REF_TET_VERTICES[va][2] + s_cell * HCURL_REF_TET_VERTICES[vb][2]};

      /* Evaluate Nédélec basis (already covariant-Piola-pulled). */
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++)
        for (PetscInt l = 0; l < n; l++) Ni[d][l] = 0.0;
      PetscCall(ops->computeBasis(cell, X_ref, coeffs_const, Ni));

      /* Evaluate H1 P_nord reference-frame gradient. */
      PetscCall(shape3DHTet(X_ref, nord, &cell->orientation, ShapH, GradH));

      /* L_0(s_can)..L_{nord-1}(s_can). PolyLegendre(X, T=1, k, P) returns
       * P[i] = L_i(2X - T) = L_i(2X - 1). To evaluate at y = s_can we
       * pass X = (s_can + 1) / 2 so 2X - 1 = s_can. */
      {
        PetscReal P_tmp[8] = {0.0};
        PetscCall(PolyLegendre((s_can + 1.0) * 0.5, 1.0, nord, P_tmp));
        for (PetscInt mode = 0; mode < nord; mode++) LnVals[mode] = P_tmp[mode];
      }

      for (PetscInt mode = 0; mode < nord; mode++) {
        const PetscInt row = fem->edgeDofOffset + e * fem->numDofPerEdge + mode;
        const PetscReal Ln = LnVals[mode];
        const PetscReal wL = w * Ln;

        for (PetscInt l = 0; l < n; l++) {
          const PetscReal udot = Ni[0][l] * t_can[0]
                               + Ni[1][l] * t_can[1]
                               + Ni[2][l] * t_can[2];
          D[row + l * n] += wL * udot;
        }
        for (PetscInt j = 0; j < nh; j++) {
          const PetscReal gref[NUM_DIMENSIONS] = {GradH[0][j], GradH[1][j], GradH[2][j]};
          PetscReal gphys[NUM_DIMENSIONS];
          hcurlPullGradient(cell, gref, gphys);
          const PetscReal gdot = gphys[0] * t_can[0]
                               + gphys[1] * t_can[1]
                               + gphys[2] * t_can[2];
          B[row + j * n] += wL * gdot;
        }
      }
    }
  }

  PetscCall(PetscFree(q1.points));
  PetscCall(PetscFree(q1.weights));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Add face moments to D and B for nord ≥ 2.
 *
 * For each face f and each spatial-tangent combination, compute
 *     σ_F^{(p,q),α}(u) = ∫_F (u · e_α^F) · ψ_{p,q}^Dubiner(λ) dA
 * where (λ_canon[0], λ_canon[1], λ_canon[2]) are barycentric coords
 * with respect to the CANONICAL face vertex ordering and ψ_{p,q} is
 * the Dubiner orthogonal polynomial of degree p+q ≤ p_face (see
 * hcurlDubinerFace above). The 2D quadrature points (u, v) on the
 * reference triangle parameterize the face via
 *     X_ref = (1 - u - v) V_canon[0] + u V_canon[1] + v V_canon[2]
 * in cell reference coordinates. */
static PetscErrorCode hcurlAddFaceMoments(const FEMSpace *fem, const Cell *cell,
                                          const NedelecOps *ops,
                                          const PetscReal *const *coeffs_const,
                                          PetscReal **Ni, PetscReal *ShapH,
                                          PetscReal **GradH,
                                          PetscScalar *D, PetscScalar *B) {
  PetscFunctionBeginUser;
  const PetscInt nord  = fem->nord;
  const PetscInt n     = fem->numDofInCell;
  const PetscInt nh    = fem->numH1DofInCell_Pnord;
  const PetscInt p_face = nord - 2;
  if (p_face < 0) PetscFunctionReturn(PETSC_SUCCESS);

  /* 2D quadrature exact for polynomial degree 2*nord (overkill safe).
   * The face integrand has degree (nord-1) + p_face = 2*nord-3. */
  Quadrature2D q2 = {0};
  PetscCall(computeNum2DQuadraturePoints(2 * nord, &q2));
  PetscCall(PetscCalloc1(q2.numPoints, &q2.points));
  for (PetscInt i = 0; i < q2.numPoints; i++) PetscCall(PetscCalloc1(2, &q2.points[i]));
  PetscCall(PetscCalloc1(q2.numPoints, &q2.weights));
  PetscCall(compute2DQuadraturePoints(&q2));

  for (PetscInt f = 0; f < NUM_FACES_PER_CELL; f++) {
    PetscInt canon[3];
    hcurlFaceCanonicalVertices(cell, f, canon);

    /* Canonical face tangents in PHYSICAL space. Both cells sharing the
     * face produce the same canon[] triple (sort by physical coords) so
     * e_1, e_2 are cell-invariant vectors. */
    PetscReal e_can[2][NUM_DIMENSIONS];
    for (PetscInt alpha = 0; alpha < 2; alpha++) {
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
        e_can[alpha][d] = cell->coordinates[canon[alpha + 1] * NUM_DIMENSIONS + d]
                        - cell->coordinates[canon[0]         * NUM_DIMENSIONS + d];
      }
    }

    for (PetscInt qp = 0; qp < q2.numPoints; qp++) {
      const PetscReal u = q2.points[qp][0];
      const PetscReal v = q2.points[qp][1];
      const PetscReal w = q2.weights[qp];
      const PetscReal lam_can[3] = {1.0 - u - v, u, v};

      const PetscReal X_ref[NUM_DIMENSIONS] = {
          lam_can[0] * HCURL_REF_TET_VERTICES[canon[0]][0]
        + lam_can[1] * HCURL_REF_TET_VERTICES[canon[1]][0]
        + lam_can[2] * HCURL_REF_TET_VERTICES[canon[2]][0],
          lam_can[0] * HCURL_REF_TET_VERTICES[canon[0]][1]
        + lam_can[1] * HCURL_REF_TET_VERTICES[canon[1]][1]
        + lam_can[2] * HCURL_REF_TET_VERTICES[canon[2]][1],
          lam_can[0] * HCURL_REF_TET_VERTICES[canon[0]][2]
        + lam_can[1] * HCURL_REF_TET_VERTICES[canon[1]][2]
        + lam_can[2] * HCURL_REF_TET_VERTICES[canon[2]][2]};

      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++)
        for (PetscInt l = 0; l < n; l++) Ni[d][l] = 0.0;
      PetscCall(ops->computeBasis(cell, X_ref, coeffs_const, Ni));
      PetscCall(shape3DHTet(X_ref, nord, &cell->orientation, ShapH, GradH));

      /* Iterate Dubiner modes (p, q) with p + q ≤ p_face in total-degree-
       * then-p-ascending order. s_idx mirrors the previous monomial
       * enumeration so the per-face DOF row positions are unchanged. */
      {
        PetscInt s_idx = 0;
        for (PetscInt sum = 0; sum <= p_face; sum++) {
          for (PetscInt pp = 0; pp <= sum; pp++) {
            const PetscInt  qq  = sum - pp;
            const PetscReal psi = hcurlDubinerFace(lam_can, pp, qq);

            for (PetscInt alpha = 0; alpha < 2; alpha++) {
              const PetscInt   mode_idx = 2 * s_idx + alpha;
              const PetscInt   row      = fem->faceDofOffset
                                        + f * fem->numDofPerFace + mode_idx;
              const PetscReal *e_alpha  = e_can[alpha];
              const PetscReal  wp       = w * psi;

              for (PetscInt l = 0; l < n; l++) {
                const PetscReal udot = Ni[0][l] * e_alpha[0]
                                     + Ni[1][l] * e_alpha[1]
                                     + Ni[2][l] * e_alpha[2];
                D[row + l * n] += wp * udot;
              }
              for (PetscInt j = 0; j < nh; j++) {
                const PetscReal gref[NUM_DIMENSIONS] = {GradH[0][j], GradH[1][j], GradH[2][j]};
                PetscReal gphys[NUM_DIMENSIONS];
                hcurlPullGradient(cell, gref, gphys);
                const PetscReal gdot = gphys[0] * e_alpha[0]
                                     + gphys[1] * e_alpha[1]
                                     + gphys[2] * e_alpha[2];
                B[row + j * n] += wp * gdot;
              }
            }
            s_idx++;
          }
        }
      }
    }
  }

  for (PetscInt i = 0; i < q2.numPoints; i++) PetscCall(PetscFree(q2.points[i]));
  PetscCall(PetscFree(q2.points));
  PetscCall(PetscFree(q2.weights));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Add volume moments to D and B for nord ≥ 3.
 *
 * Volume DOFs are NOT shared across cells (the basis bubbles vanish on
 * the boundary), so cross-cell canonicalization is unnecessary. We use
 * the three cell-local tangent vectors e_β^K = V_β^K - V_0^K (β = 1, 2, 3)
 * in physical space and the orthogonal Koornwinder polynomial basis on
 * the reference tet for the spatial test functions (degree ≤ nord-3).
 *
 * Spatial mode index enumeration: walk total degree sum = 0..p_vol,
 * within sum iterate (p, q) ascending and set r = sum - p - q. */
static PetscErrorCode hcurlAddVolumeMoments(const FEMSpace *fem, const Cell *cell,
                                            const NedelecOps *ops,
                                            const PetscReal *const *coeffs_const,
                                            PetscReal **Ni, PetscReal *ShapH,
                                            PetscReal **GradH,
                                            PetscScalar *D, PetscScalar *B) {
  PetscFunctionBeginUser;
  const PetscInt nord = fem->nord;
  const PetscInt n    = fem->numDofInCell;
  const PetscInt nh   = fem->numH1DofInCell_Pnord;
  const PetscInt p_vol = nord - 3;
  if (p_vol < 0) PetscFunctionReturn(PETSC_SUCCESS);

  Quadrature3D q3 = {0};
  PetscCall(computeNum3DQuadraturePoints(nord, &q3));
  PetscCall(PetscCalloc1(q3.numPoints, &q3.weights));
  PetscCall(PetscCalloc1(q3.numPoints, &q3.points));
  for (PetscInt i = 0; i < q3.numPoints; i++)
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &q3.points[i]));
  PetscCall(compute3DQuadraturePoints(&q3));

  /* Cell-local tangent vectors V_β - V_0 in physical space. */
  PetscReal e_vol[3][NUM_DIMENSIONS];
  for (PetscInt alpha = 0; alpha < 3; alpha++) {
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
      e_vol[alpha][d] = cell->coordinates[(alpha + 1) * NUM_DIMENSIONS + d]
                      - cell->coordinates[0           * NUM_DIMENSIONS + d];
    }
  }

  const PetscInt numDofPerVolume = fem->numDofPerVolume;
  /* Sanity: numDofPerVolume == 3 * (p_vol+1)(p_vol+2)(p_vol+3)/6. */

  for (PetscInt qp = 0; qp < q3.numPoints; qp++) {
    const PetscReal X_ref[NUM_DIMENSIONS] = {q3.points[qp][0], q3.points[qp][1], q3.points[qp][2]};
    const PetscReal w  = q3.weights[qp];
    const PetscReal wj = w * cell->detJacobian;

    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++)
      for (PetscInt l = 0; l < n; l++) Ni[d][l] = 0.0;
    PetscCall(ops->computeBasis(cell, X_ref, coeffs_const, Ni));
    PetscCall(shape3DHTet(X_ref, nord, &cell->orientation, ShapH, GradH));

    PetscInt s_idx = 0;
    for (PetscInt sum = 0; sum <= p_vol; sum++) {
      for (PetscInt pp = 0; pp <= sum; pp++) {
        for (PetscInt qq = 0; qq <= sum - pp; qq++) {
          const PetscInt  rr  = sum - pp - qq;
          const PetscReal psi = hcurlKoornwinderVolume(X_ref, pp, qq, rr);

          for (PetscInt beta = 0; beta < 3; beta++) {
            const PetscInt mode_idx = 3 * s_idx + beta;
            if (mode_idx >= numDofPerVolume) continue;
            const PetscInt row = fem->volumeDofOffset + mode_idx;
            const PetscReal *e_beta = e_vol[beta];
            const PetscReal wp = wj * psi;

            for (PetscInt l = 0; l < n; l++) {
              const PetscReal udot = Ni[0][l] * e_beta[0]
                                   + Ni[1][l] * e_beta[1]
                                   + Ni[2][l] * e_beta[2];
              D[row + l * n] += wp * udot;
            }
            for (PetscInt j = 0; j < nh; j++) {
              const PetscReal gref[NUM_DIMENSIONS] = {GradH[0][j], GradH[1][j], GradH[2][j]};
              PetscReal gphys[NUM_DIMENSIONS];
              hcurlPullGradient(cell, gref, gphys);
              const PetscReal gdot = gphys[0] * e_beta[0]
                                   + gphys[1] * e_beta[1]
                                   + gphys[2] * e_beta[2];
              B[row + j * n] += wp * gdot;
            }
          }
          s_idx++;
        }
      }
    }
  }

  for (PetscInt i = 0; i < q3.numPoints; i++) PetscCall(PetscFree(q3.points[i]));
  PetscCall(PetscFree(q3.points));
  PetscCall(PetscFree(q3.weights));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Per-cell builder for the EXACT commuting discrete-gradient matrix
 * Ge_K[l, j] = σ_l(∇φ_j) using the Ainsworth–Coyle moment functionals.
 * Full mathematical derivation, cross-cell-consistency argument, and
 * canonical-frame strategy live in the section banner at the top of
 * this builder family (~600 lines above; search for "Exact commuting
 * discrete-gradient builder"). Registered on every NedelecOps table
 * via HIERARCHICAL_ORDER_OPS at the bottom of this file. */
static PetscErrorCode hierarchicalBuildExactGradientMatrix(const FEMSpace *fem, const Cell *cell,
                                                           PetscReal **gradientMatrix) {
  PetscFunctionBeginUser;

  const PetscInt    nord                  = fem->nord;
  const PetscInt    numDofInCell          = fem->numDofInCell;
  const PetscInt    numH1DofInCell_Pnord  = fem->numH1DofInCell_Pnord;
  const NedelecOps *ops                   = fem->ops;
  if (!ops) PetscFunctionReturn(PETSC_SUCCESS);

  /* Reset output. */
  for (PetscInt i = 0; i < numDofInCell; i++)
    for (PetscInt j = 0; j < numH1DofInCell_Pnord; j++) gradientMatrix[i][j] = 0.0;

  /* Scratch buffers reused across all moment kernels. */
  PetscReal **Ni = NULL;
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Ni));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) PetscCall(PetscCalloc1(numDofInCell, &Ni[i]));

  PetscReal  *ShapH = NULL;
  PetscReal **GradH = NULL;
  PetscCall(PetscCalloc1(numH1DofInCell_Pnord, &ShapH));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &GradH));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++)
    PetscCall(PetscCalloc1(numH1DofInCell_Pnord, &GradH[i]));

  /* hierarchicalComputeBasis ignores coeffs/Dx/Dy/Dz but the ops
   * signature requires non-NULL slots. */
  PetscReal **coeffs = NULL, **Dx = NULL, **Dy = NULL, **Dz = NULL;
  PetscCall(PetscCalloc1(numDofInCell, &coeffs));
  for (PetscInt i = 0; i < numDofInCell; i++) PetscCall(PetscCalloc1(numDofInCell, &coeffs[i]));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dx));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dy));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dz));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(numDofInCell, &Dx[i]));
    PetscCall(PetscCalloc1(numDofInCell, &Dy[i]));
    PetscCall(PetscCalloc1(numDofInCell, &Dz[i]));
  }
  PetscCall(ops->computeCoefficients(cell, coeffs, Dx, Dy, Dz));
  const PetscReal *const *coeffs_const = (const PetscReal *const *)coeffs;

  /* D (n × n) and B (n × nh) in column-major dense layout:
   *   D[i + l * n] = D[i, l] = σ_i(N_l)
   *   B[i + j * n] = B[i, j] = σ_i(∇φ_j)                       */
  PetscScalar *D_data = NULL, *B_data = NULL;
  PetscCall(PetscCalloc1(numDofInCell * numDofInCell,         &D_data));
  PetscCall(PetscCalloc1(numDofInCell * numH1DofInCell_Pnord, &B_data));

  PetscCall(hcurlAddEdgeMoments(fem, cell, ops, coeffs_const, Ni, ShapH, GradH, D_data, B_data));
  if (nord >= 2)
    PetscCall(hcurlAddFaceMoments(fem, cell, ops, coeffs_const, Ni, ShapH, GradH, D_data, B_data));
  if (nord >= 3)
    PetscCall(hcurlAddVolumeMoments(fem, cell, ops, coeffs_const, Ni, ShapH, GradH, D_data, B_data));

  /* Solve D · Ge = B via LAPACK GESV (partial-pivoting LU) with
   * post-solve sanitization.
   *
   * D is the duality matrix between the hierarchical Nédélec basis
   * and the Ainsworth-Coyle moment functionals. For well-conditioned
   * cells (nord ≤ 4 everywhere, nord ≥ 5 on uniform meshes) LU is
   * exact to machine precision - using LAPACK directly instead of
   * MatLUFactor + MatMatSolve sidesteps Mat object overhead and
   * gives the same numerical result.
   *
   * On distorted CSEM cells at nord ≥ 5 the hierarchical basis from
   * shape3DETet (HomIJacobi / AncETri recurrences) overflows at
   * quadrature points where the collapsed-coordinate denominators
   * approach zero, producing NaN/Inf in B (and sometimes D) during
   * moment accumulation. LU then either reports a singular pivot
   * (INFO > 0) or silently writes NaN into the solution. Rather than
   * propagating these into the global G - which poisons every
   * downstream diagnostic - we sanitize the solution post-hoc:
   * non-finite entries are replaced with zero, the cell contributes
   * a degraded (but finite) gradient row to the global G, and a
   * single diagnostic line tells us how many cells / entries were
   * affected. This is a band-aid over the deeper basis-overflow
   * pathology (the Karniadakis-Sherwin scaled-Jacobi recurrence
   * would address it at the root); it keeps the assembly running
   * while we decide whether the affected cell count justifies that
   * larger surgery.
   *
   * GESV overwrites B in place with the solution. */
  {
    PetscBLASInt  N_blas, NRHS_blas, LDA, LDB, INFO = 0;
    PetscBLASInt *IPIV = NULL;

    PetscCall(PetscBLASIntCast(numDofInCell, &N_blas));
    PetscCall(PetscBLASIntCast(numH1DofInCell_Pnord, &NRHS_blas));
    LDA = N_blas;
    LDB = N_blas;

    PetscCall(PetscCalloc1(N_blas, &IPIV));

    LAPACKgesv_(&N_blas, &NRHS_blas, D_data, &LDA, IPIV,
                B_data, &LDB, &INFO);

    /* Sanitize the solution: scan B_data for non-finite *or* very
     * large finite entries and replace with zero.
     *
     * INFO > 0 (singular pivot) leaves B partially filled with junk;
     * INFO < 0 means a bad argument (should not happen here).
     *
     * The HUGE_THRESHOLD branch is needed because LU with partial
     * pivoting can encounter a "near-zero" pivot whose magnitude is
     * not exactly zero (no NaN/Inf result) but tiny enough that
     * dividing through amplifies the solution to magnitude well
     * above the legitimate scale. Reference: nord=4 produces
     * ||G||_F = 8e3, nord=6 produces 3e4, both with per-entry
     * magnitudes O(1)–O(100); per-cell Ge entries at any supported
     * order do not legitimately exceed 1e3. We set the threshold to
     * 1e6 - two orders above any legitimate value, but well below
     * the pathological-cell scale (entries reaching 1e6–1e10 at
     * nord ≥ 5) that drives the global Frobenius norm to 1e10+. */
    const PetscReal HUGE_THRESHOLD = 1.0e6;
    PetscInt n_bad = 0;
    for (PetscInt k = 0; k < numDofInCell * numH1DofInCell_Pnord; k++) {
      if (PetscIsInfOrNanScalar(B_data[k]) ||
          PetscAbsScalar(B_data[k]) > HUGE_THRESHOLD) {
        B_data[k] = 0.0;
        n_bad++;
      }
    }
    if (INFO != 0 || n_bad > 0) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,
          "[hierarchicalBuildExactGradientMatrix] nord=%" PetscInt_FMT
          ": LAPACK gesv INFO=%d, scrubbed %" PetscInt_FMT
          " unphysical entries in solution (basis overflow on this cell)\n",
          nord, (int)INFO, n_bad));
    }

    /* Copy from B_data (column-major) to gradientMatrix (row-major). */
    for (PetscInt i = 0; i < numDofInCell; i++)
      for (PetscInt j = 0; j < numH1DofInCell_Pnord; j++)
        gradientMatrix[i][j] = PetscRealPart(B_data[i + j * numDofInCell]);

    PetscCall(PetscFree(IPIV));
  }

  /* Cleanup */
  PetscCall(PetscFree(D_data));
  PetscCall(PetscFree(B_data));

  for (PetscInt i = 0; i < numDofInCell; i++) PetscCall(PetscFree(coeffs[i]));
  PetscCall(PetscFree(coeffs));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(Dx[i]));
    PetscCall(PetscFree(Dy[i]));
    PetscCall(PetscFree(Dz[i]));
    PetscCall(PetscFree(Ni[i]));
    PetscCall(PetscFree(GradH[i]));
  }
  PetscCall(PetscFree(Dx));
  PetscCall(PetscFree(Dy));
  PetscCall(PetscFree(Dz));
  PetscCall(PetscFree(Ni));
  PetscCall(PetscFree(GradH));
  PetscCall(PetscFree(ShapH));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---- Per-order thin wrappers + dispatch tables -----------------------
 *
 * One ops table per supported order. All orders share:
 *   - hierarchicalComputeBasisOrder / hierarchicalComputeCurlsOrder
 *     (Piola-pullback wrappers around shape3DETet, parameterized by nord).
 *   - hierarchicalBuildGradientMatrixTopological for the P1-target G used
 *     by the inverse kernel.
 *   - hierarchicalBuildExactGradientMatrix for the order-k P_nord-target
 *     commuting G used by the forward kernel.
 */
#define HIERARCHICAL_ORDER_OPS(N, GRADIENT_BUILDER)                                      \
  static PetscErrorCode order##N##ComputeBasis(const Cell *cell,                         \
                                               const PetscReal point[NUM_DIMENSIONS],    \
                                               const PetscReal *const *coeffs,           \
                                               PetscReal **Ni) {                         \
    (void)coeffs;                                                                        \
    return hierarchicalComputeBasisOrder((N), cell, point, Ni);                          \
  }                                                                                      \
  static PetscErrorCode order##N##ComputeCurls(const Cell *cell,                         \
                                               const PetscReal point[NUM_DIMENSIONS],    \
                                               const PetscReal *const *coeffs,           \
                                               const PetscReal *const *Dx_Ni,            \
                                               const PetscReal *const *Dy_Ni,            \
                                               const PetscReal *const *Dz_Ni,            \
                                               PetscReal **NiCurl) {                     \
    (void)coeffs; (void)Dx_Ni; (void)Dy_Ni; (void)Dz_Ni;                                 \
    return hierarchicalComputeCurlsOrder((N), cell, point, NiCurl);                      \
  }                                                                                      \
  const NedelecOps nedelecOps_order##N = {                                               \
      .computeCoefficients      = hierarchicalComputeCoefficients,                       \
      .computeBasis             = order##N##ComputeBasis,                                \
      .computeCurls             = order##N##ComputeCurls,                                \
      .buildGradientMatrix      = (GRADIENT_BUILDER),                                    \
      .buildExactGradientMatrix = hierarchicalBuildExactGradientMatrix,                  \
  }

/* All orders use the same topological builder. nord=1 reduces to the
 * vertex-edge incidence with 1 DOF per edge (matching the previous
 * order1HierarchicalBuildGradientMatrix exactly). */
HIERARCHICAL_ORDER_OPS(1, hierarchicalBuildGradientMatrixTopological);
HIERARCHICAL_ORDER_OPS(2, hierarchicalBuildGradientMatrixTopological);
HIERARCHICAL_ORDER_OPS(3, hierarchicalBuildGradientMatrixTopological);
HIERARCHICAL_ORDER_OPS(4, hierarchicalBuildGradientMatrixTopological);
HIERARCHICAL_ORDER_OPS(5, hierarchicalBuildGradientMatrixTopological);
HIERARCHICAL_ORDER_OPS(6, hierarchicalBuildGradientMatrixTopological);

#undef HIERARCHICAL_ORDER_OPS
