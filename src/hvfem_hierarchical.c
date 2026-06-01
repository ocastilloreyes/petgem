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
 * Discrete-gradient builder (all orders):
 *   - TOPOLOGICAL (hierarchicalBuildGradientMatrixTopological): vertex-
 *     edge incidence. Lowest-order Whitney slot per mesh edge maps to
 *     that edge's 2 endpoint vertex H1 DOFs; face/volume / higher-order
 *     edge DOF rows are zero. Consumed by PCBDDCSetDiscreteGradient as
 *     the curl-kernel hint at every order.
 */

#include <petsc.h>
#include <petscsys.h>

#include "constants.h"
#include "hvfem.h"
#include "hvfem_internal.h"

/* ---------------------------------------------------------------------------
 * Local helpers - small wrappers over hvfem.c machinery exposed via
 * hvfem_internal.h. AncEE / AncETri are unique to the hierarchical basis
 * and live here as file-local statics.
 * ------------------------------------------------------------------------- */

/**
 * @brief Evaluates homogenized Legendre polynomials in two variables.
 *
 * Thin wrapper around PolyLegendre (declared in hvfem_internal.h).
 *
 * @param[in]  S     Homogeneous coordinate pair (S[0], S[1]).
 * @param[in]  nord  Highest polynomial order.
 * @param[out] HomP  Polynomial values P_0..P_nord.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode HomLegendre(const PetscReal S[2], const PetscInt nord, PetscReal HomP[]) {
  PetscFunctionBeginUser;
  PetscCall(PolyLegendre(S[1], S[0] + S[1], nord, HomP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Edge-block ancillary functions for the hierarchical Nédélec basis.
 *
 * Returns EE[3][nord] (vector value) and CurlEE[3][nord] (curl) on the
 * reference cell. When Idec is PETSC_TRUE the function collapses to a
 * degenerate edge whose Whitney function vanishes; in that case the curls
 * are zero.
 *
 * @param[in]  S       Edge-projected coordinate pair.
 * @param[in]  DS      Gradients of the projected coordinates.
 * @param[in]  nord    Polynomial order.
 * @param[in]  Idec    Decoupled-coordinate (degenerate-edge) flag.
 * @param[out] EE      Edge basis values (3 × nord).
 * @param[out] CurlEE  Edge basis curls (3 × nord).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
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

/**
 * @brief Face-block ancillary functions for the hierarchical Nédélec basis.
 *
 * Builds ETri[3][nord-1][nord-1] and CurlETri[3][nord-1][nord-1] by
 * combining AncEE on the face's tangent coordinates with HomIJacobi in the
 * third coordinate.
 *
 * @param[in]  S         Face-projected coordinate triple.
 * @param[in]  DS        Gradients of the projected coordinates.
 * @param[in]  nord      Polynomial order.
 * @param[in]  Idec      Decoupled-coordinate flag.
 * @param[out] ETri      Face basis values (3 × (nord-1) × (nord-1)).
 * @param[out] CurlETri  Face basis curls (3 × (nord-1) × (nord-1)).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
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

/**
 * @brief Evaluates the full hierarchical H(curl) Nédélec basis on the
 *        reference tetrahedron.
 *
 * The columns are returned in PETSc DOF order (matching what the rest of
 * the assembly expects). Conversion to physical space (covariant Piola for
 * values, curl Piola for curls) is performed by the per-order adapters
 * (hierarchicalComputeBasisOrder / hierarchicalComputeCurlsOrder).
 *
 * @param[in]  X                Reference-cell point (ξ, η, ζ).
 * @param[in]  nord             Polynomial order (1..6).
 * @param[in]  cellOrientation  4 face codes (PETGEM convention, 0..5) and 6
 *                              edge signs from DMPlex (±1).
 * @param[out] ShapE            Reference-cell vector value of each shape
 *                              function (NUM_DIMENSIONS × numDofInCell).
 * @param[out] CurlE            Reference-cell curl of each shape function
 *                              (NUM_DIMENSIONS × numDofInCell).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
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

/**
 * @brief No-op coefficient builder for the hierarchical Nédélec basis.
 *
 * The hierarchical basis evaluates directly inside shape3DETet, so there
 * are no per-cell precomputed coefficients to populate. All arguments are
 * ignored; provided so the ops table has a uniform signature.
 *
 * @param[in]  cell    Cell descriptor (unused).
 * @param[out] coeffs  Coefficient buffer (unused).
 * @param[out] Dx_Ni   ∂/∂x derivative buffer (unused).
 * @param[out] Dy_Ni   ∂/∂y derivative buffer (unused).
 * @param[out] Dz_Ni   ∂/∂z derivative buffer (unused).
 *
 * @return PETSC_SUCCESS always.
 */
static PetscErrorCode hierarchicalComputeCoefficients(const Cell *cell, PetscReal **coeffs,
                                                      PetscReal **Dx_Ni, PetscReal **Dy_Ni,
                                                      PetscReal **Dz_Ni) {
  (void)cell; (void)coeffs; (void)Dx_Ni; (void)Dy_Ni; (void)Dz_Ni;
  return PETSC_SUCCESS;
}

/**
 * @brief Shared adapter that evaluates Nédélec basis values at a point.
 *
 * Parameterized by `nord`, called by the per-order thin wrappers below.
 * Calls shape3DETet on the reference cell and applies the covariant Piola
 * pullback Ni_phys = J^{-T}·Ni_ref using cell->invJacobian.
 *
 * @param[in]  nord   Polynomial order.
 * @param[in]  cell   Cell with computed jacobian / orientation.
 * @param[in]  point  Reference-cell evaluation point.
 * @param[out] Ni     Basis values (NUM_DIMENSIONS × numDofInCell).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
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

/**
 * @brief Shared adapter that evaluates Nédélec basis curls at a point.
 *
 * Calls shape3DETet on the reference cell and applies the curl Piola
 * pullback curl_phys = (J · curl_ref) / det(J).
 *
 * @param[in]  nord    Polynomial order.
 * @param[in]  cell    Cell with computed jacobian / orientation.
 * @param[in]  point   Reference-cell evaluation point.
 * @param[out] NiCurl  Basis curls (NUM_DIMENSIONS × numDofInCell).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
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

/**
 * @brief Builds the topological discrete-gradient matrix (lowest-Whitney slot).
 *
 * Maps each mesh edge's lowest-Whitney slot to its two endpoint vertex H1
 * DOFs; consumed by PCBDDCSetDiscreteGradient as the curl-kernel hint at
 * every order. Only the LOWEST-ORDER Whitney DOF per mesh edge carries the
 * ±1 vertex incidence; higher-order edge DOFs (n > 0) and all face/volume
 * DOF rows are zero.
 *
 * Sign convention: edges in canonical orientation (NoriE=0, edgeSigns ≥ 0)
 * get -1 at the start vertex and +1 at the end vertex of EDGE_VERTICES[e].
 * Reversed edges get the opposite - the standard Whitney tangential
 * direction.
 *
 * @param[in]  fem             Finite-element space descriptor.
 * @param[in]  cell            Cell with computed orientation.
 * @param[in]  quadrature1d    1D quadrature (unused; signature compatibility).
 * @param[out] gradientMatrix  Per-cell gradient block
 *                             (numDofInCell × numH1DofInCell).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
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

/* ---- Per-order thin wrappers + dispatch tables -----------------------
 *
 * One ops table per supported order. All orders share:
 *   - hierarchicalComputeBasisOrder / hierarchicalComputeCurlsOrder
 *     (Piola-pullback wrappers around shape3DETet, parameterized by nord).
 *   - hierarchicalBuildGradientMatrixTopological for the lowest-Whitney
 *     vertex-incidence G used as the PCBDDC discrete-gradient hint.
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
