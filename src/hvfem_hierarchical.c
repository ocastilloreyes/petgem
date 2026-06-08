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
 *   - buildExactDiscreteGradient: the EXACT order-p discrete gradient against
 *     the full P_nord H1 closure (grad(phi_k) = sum_i G_ik N_i, so K·G = 0).
 *     Consumed by PCBDDCSetDiscreteGradient as the curl-kernel coarse-space
 *     operator. verifyExactDiscreteGradientCell unit-checks it (-fm_check_gradient).
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
 * @brief Solves the SPD system A X = B by Cholesky + iterative refinement.
 *
 * Replaces the earlier explicit Gauss-Jordan inverse (invertMatrixDense): forming
 * M^{-1} and multiplying squares the conditioning damage, which is what made the
 * nord=5 reference Nédélec mass solve collapse to a 1e-3 residual on the wham
 * case. A here is the reference Nédélec mass matrix M, symmetric positive
 * definite, so a Cholesky factor A = L L^T plus a few refinement sweeps is both
 * faster (O(N^3/3) vs the inverse) and markedly more accurate.
 *
 * Two diagnostics are returned to separate the failure modes:
 *   - relResidual = ||B - A X||_F / ||B||_F. Refinement drives the BACKWARD error
 *     (this residual) toward machine eps even when A is ill-conditioned, so a
 *     small value only confirms the solve converged; it does NOT prove X is
 *     accurate.
 *   - condEst = max/min Cholesky pivot d_j (an SPD lower bound on kappa_2(A)).
 *     The FORWARD error of X is ~ condEst * eps, so this is the metric that
 *     decides whether a large downstream gradient residual is a conditioning
 *     artifact (condEst huge) or a genuine basis fault (condEst modest).
 *
 * Not capped at N <= 6 (unlike invertMatrix in hvfem.c): N = numDofInCell, up to
 * 216 at nord=6.
 *
 * @param[in]  N            System dimension.
 * @param[in]  A            Row-major N x N SPD matrix (preserved).
 * @param[in]  nrhs         Number of right-hand sides (columns of B/X).
 * @param[in]  Bmat         Row-major N x nrhs right-hand sides.
 * @param[out] Xmat         Row-major N x nrhs solution.
 * @param[out] relResidual  ||B - A X||_F / ||B||_F (may be NULL).
 * @param[out] condEst      max/min Cholesky pivot ratio (may be NULL).
 *
 * @return PetscErrorCode PETSC_SUCCESS, or PETSC_ERR_MAT_CH_ZRPVT if A is not
 *         numerically SPD (a Cholesky breakdown is itself a strong diagnostic:
 *         basis degeneracy or extreme ill-conditioning).
 */
static PetscErrorCode choleskySolveSPDRefined(PetscInt N, const PetscReal *A, PetscInt nrhs,
                                              const PetscReal *Bmat, PetscReal *Xmat,
                                              PetscReal *relResidual, PetscReal *condEst) {
  PetscFunctionBeginUser;
  const PetscInt nRefine = 2;          /* refinement sweeps after the initial solve */
  PetscReal *L, *b, *x, *y;
  PetscReal  dMin = PETSC_MAX_REAL, dMax = 0.0;
  PetscCall(PetscCalloc1(N * N, &L));
  PetscCall(PetscCalloc1(N, &b));
  PetscCall(PetscCalloc1(N, &x));
  PetscCall(PetscCalloc1(N, &y));

  /* Cholesky factor A = L L^T (lower triangle). */
  for (PetscInt j = 0; j < N; j++) {
    PetscReal d = A[j * N + j];
    for (PetscInt k = 0; k < j; k++) d -= L[j * N + k] * L[j * N + k];
    if (d <= 0.0) {
      PetscCall(PetscFree(L));
      PetscCall(PetscFree(b));
      PetscCall(PetscFree(x));
      PetscCall(PetscFree(y));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT,
              "Cholesky breakdown at pivot %" PetscInt_FMT " (d=%g): reference Nédélec mass "
              "matrix is not numerically SPD (basis degeneracy or extreme ill-conditioning, N=%" PetscInt_FMT ")",
              j, (double)d, N);
    }
    dMin = PetscMin(dMin, d);
    dMax = PetscMax(dMax, d);
    L[j * N + j] = PetscSqrtReal(d);
    for (PetscInt i = j + 1; i < N; i++) {
      PetscReal s = A[i * N + j];
      for (PetscInt k = 0; k < j; k++) s -= L[i * N + k] * L[j * N + k];
      L[i * N + j] = s / L[j * N + j];
    }
  }
  if (condEst) *condEst = (dMin > 0.0) ? dMax / dMin : PETSC_MAX_REAL;

  PetscReal resNum = 0.0, resDen = 0.0;
  for (PetscInt c = 0; c < nrhs; c++) {
    for (PetscInt i = 0; i < N; i++) { b[i] = Bmat[i * nrhs + c]; x[i] = 0.0; }

    /* Initial solve (x = 0 -> first residual is b) followed by nRefine sweeps.
     * Each sweep: r = b - A x; solve A dx = r via L; x += dx. */
    for (PetscInt it = 0; it <= nRefine; it++) {
      for (PetscInt i = 0; i < N; i++) {       /* y <- r = b - A x */
        PetscReal s = b[i];
        for (PetscInt k = 0; k < N; k++) s -= A[i * N + k] * x[k];
        y[i] = s;
      }
      for (PetscInt i = 0; i < N; i++) {       /* forward: L z = r, z over y */
        PetscReal s = y[i];
        for (PetscInt k = 0; k < i; k++) s -= L[i * N + k] * y[k];
        y[i] = s / L[i * N + i];
      }
      for (PetscInt i = N - 1; i >= 0; i--) {  /* back: L^T dx = z, dx over y */
        PetscReal s = y[i];
        for (PetscInt k = i + 1; k < N; k++) s -= L[k * N + i] * y[k];
        y[i] = s / L[i * N + i];
      }
      for (PetscInt i = 0; i < N; i++) x[i] += y[i];
    }

    for (PetscInt i = 0; i < N; i++) Xmat[i * nrhs + c] = x[i];

    for (PetscInt i = 0; i < N; i++) {         /* accumulate ||b - A x||, ||b|| */
      PetscReal s = b[i];
      for (PetscInt k = 0; k < N; k++) s -= A[i * N + k] * x[k];
      resNum += s * s;
      resDen += b[i] * b[i];
    }
  }
  if (relResidual) *relResidual = (resDen > 0.0) ? PetscSqrtReal(resNum / resDen) : 0.0;

  PetscCall(PetscFree(L));
  PetscCall(PetscFree(b));
  PetscCall(PetscFree(x));
  PetscCall(PetscFree(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the EXACT high-order discrete gradient block for one cell.
 *
 * Replaces the topological order-1 incidence with the full discrete de Rham
 * gradient PCBDDC requires for high-order Nédélec: for each P_nord H1 basis
 * function phi_k it expresses grad(phi_k) in the Nédélec basis,
 *     grad(phi_k) = sum_i C_ik N_i,     gradientMatrix[i][k] = C_ik,
 * i.e. the matrix G : S_h^p -> V_h^p that PCBDDCSetDiscreteGradient() expects.
 * It is sparse; BDDC reads the sparsity of G^T G to recover the vertex/edge/
 * face equivalence classes (per S. Zampini). The order-1 topological builder
 * leaves higher-order rows zero, which collapses the coarse space onto the
 * whole interface; the previous dense moment-gradient had the opposite fault
 * (spurious fill -> "more than two corners" / "SIZE OF EDGE > EXTCOL").
 *
 * Construction (geometry-independent): grad_ref(phi_k) lies exactly in
 * span{N_i^ref} (FEEC exact sequence), so the reference-frame L2 projection
 *     M C = B,  M_ij = <N_i^ref,N_j^ref>,  B_ik = <N_i^ref, grad_ref phi_k>
 * gives the exact coefficients (the Jacobian J^{-T} cancels between N and
 * grad). The SPD system M C = B is solved by Cholesky + iterative refinement
 * (choleskySolveSPDRefined) rather than an explicit inverse - the inverse
 * squared the conditioning damage and made the nord=5 solve diverge. A relative
 * threshold then restores the true sparse pattern - WITHOUT it, floating-point
 * fill makes G dense and corrupts BDDC's analysis (this threshold is the fix the
 * earlier implementation lacked).
 *
 * Row  i: Nédélec DOF (edge/face/volume), DMPlex closure order (= dofIndices).
 * Col  k: P_nord H1 DOF (vertex/edge/face/volume), DMPlex closure order
 *         (= shape3DHTet output order = grid.H1dm_Pnord closure).
 *
 * Orientation is baked into both bases (shape3DETet / shape3DHTet take
 * cell->orientation), so the +/- signs are correct per cell. Cost is
 * O(numDofInCell^3) per cell (a dense Cholesky factor); for high order it
 * should be cached/reused across equal-orientation cells (TODO - fine for the
 * validation target).
 *
 * @param[in]  fem             FE space (nord, numDofInCell, numH1DofInCell_Pnord).
 * @param[in]  cell            Cell with computed orientation.
 * @param[out] gradientMatrix  Per-cell block (numDofInCell x numH1DofInCell_Pnord).
 * @param[out] laResidual      ||M C - B||_F / ||B||_F of the solve (may be NULL).
 * @param[out] condEst         max/min Cholesky pivot of M (may be NULL).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode buildExactDiscreteGradient(const FEMSpace *fem, const Cell *cell,
                                          PetscReal **gradientMatrix,
                                          PetscReal *laResidual, PetscReal *condEst) {
  PetscFunctionBeginUser;
  const PetscInt nord = fem->nord;
  const PetscInt nNed = fem->numDofInCell;           /* rows: Nédélec        */
  const PetscInt nH1  = fem->numH1DofInCell_Pnord;   /* cols: P_nord H1      */

  for (PetscInt i = 0; i < nNed; i++)
    for (PetscInt k = 0; k < nH1; k++) gradientMatrix[i][k] = 0.0;

  /* Reference quadrature. computeNum3DQuadraturePoints(nord) already returns a
   * rule exact for the degree-(2*nord) mass integrand (same one the assembly
   * uses for Me), which also covers B (degree 2*nord-1). */
  Quadrature3D q;
  PetscCall(computeNum3DQuadraturePoints(nord, &q));
  PetscCall(PetscCalloc1(q.numPoints, &q.points));
  for (PetscInt qp = 0; qp < q.numPoints; qp++)
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &q.points[qp]));
  PetscCall(PetscCalloc1(q.numPoints, &q.weights));
  PetscCall(compute3DQuadraturePoints(&q));

  /* Per-point scratch: reference Nédélec values/curls, H1 values/gradients. */
  PetscReal **Eref, **Cref, *Hval, **Hgrad;
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Eref));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Cref));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Hgrad));
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    PetscCall(PetscCalloc1(nNed, &Eref[d]));
    PetscCall(PetscCalloc1(nNed, &Cref[d]));   /* curls: unused, but shape3DETet fills them */
    PetscCall(PetscCalloc1(nH1,  &Hgrad[d]));
  }
  PetscCall(PetscCalloc1(nH1, &Hval));

  /* Reference Nédélec mass matrix M (nNed x nNed), projection RHS B and the
   * coefficient block C (both nNed x nH1, row-major). */
  PetscReal *M, *B, *C;
  PetscCall(PetscCalloc1(nNed * nNed, &M));
  PetscCall(PetscCalloc1(nNed * nH1,  &B));
  PetscCall(PetscCalloc1(nNed * nH1,  &C));

  for (PetscInt qp = 0; qp < q.numPoints; qp++) {
    PetscCall(shape3DETet(q.points[qp], nord, &cell->orientation, Eref, Cref));
    PetscCall(shape3DHTet(q.points[qp], nord, &cell->orientation, Hval, Hgrad));
    const PetscReal w = q.weights[qp];
    for (PetscInt i = 0; i < nNed; i++) {
      for (PetscInt j = 0; j < nNed; j++)
        M[i * nNed + j] += w * (Eref[0][i] * Eref[0][j] + Eref[1][i] * Eref[1][j] + Eref[2][i] * Eref[2][j]);
      for (PetscInt k = 0; k < nH1; k++)
        B[i * nH1 + k] += w * (Eref[0][i] * Hgrad[0][k] + Eref[1][i] * Hgrad[1][k] + Eref[2][i] * Hgrad[2][k]);
    }
  }

  /* Exact coefficients solve M C = B (projection is exact since grad(phi_k) in
   * span{N_i}). SPD Cholesky + refinement, not an explicit inverse; the residual
   * and conditioning of M flow back to the diagnostic. */
  PetscCall(choleskySolveSPDRefined(nNed, M, nH1, B, C, laResidual, condEst));
  PetscReal maxabs = 0.0;
  for (PetscInt i = 0; i < nNed; i++)
    for (PetscInt k = 0; k < nH1; k++) {
      gradientMatrix[i][k] = C[i * nH1 + k];
      maxabs = PetscMax(maxabs, PetscAbsReal(C[i * nH1 + k]));
    }

  /* Restore the exact sparse pattern: drop floating-point fill. */
  const PetscReal tol = 1.0e-8 * maxabs;
  for (PetscInt i = 0; i < nNed; i++)
    for (PetscInt k = 0; k < nH1; k++)
      if (PetscAbsReal(gradientMatrix[i][k]) < tol) gradientMatrix[i][k] = 0.0;

  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    PetscCall(PetscFree(Eref[d]));
    PetscCall(PetscFree(Cref[d]));
    PetscCall(PetscFree(Hgrad[d]));
  }
  PetscCall(PetscFree(Eref));
  PetscCall(PetscFree(Cref));
  PetscCall(PetscFree(Hgrad));
  PetscCall(PetscFree(Hval));
  PetscCall(PetscFree(M));
  PetscCall(PetscFree(B));
  PetscCall(PetscFree(C));
  for (PetscInt qp = 0; qp < q.numPoints; qp++) PetscCall(PetscFree(q.points[qp]));
  PetscCall(PetscFree(q.points));
  PetscCall(PetscFree(q.weights));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Unit check for buildExactDiscreteGradient on one cell (no BDDC).
 *
 * Verifies the defining identity grad(phi_k) = sum_i G_ik N_i at several interior
 * reference points (max abs error), and reports - via @p result - the solve
 * health and the entity-local sparsity PCBDDC requires. Three things are
 * separated so the wham nord=5 cliff can be diagnosed without guessing:
 *   - condEst (max/min Cholesky pivot of M): a HUGE value with a large pointwise
 *     residual means the failure is conditioning of the dense projection, not the
 *     basis; a MODEST value with a large residual means a genuine basis fault.
 *   - crossEntityRatio: the Frobenius energy fraction of G living in FORBIDDEN
 *     entity blocks (edge-row coupling into face/volume H1, or face-row into
 *     volume H1). ~0 means the gradient is entity-local and the planned
 *     per-entity construction will keep it exact (gate 1b PASS).
 *   - maxEdgeRowNnz: must equal nord+1 (2 endpoint vertices + nord-1 edge
 *     bubbles); the exact per-row budget PCBDDC checks (bddcprivate.c:531).
 *
 * @param[in]  fem     FE space descriptor.
 * @param[in]  cell    Cell with computed orientation.
 * @param[out] result  Filled diagnostic metrics (must be non-NULL).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode verifyExactDiscreteGradientCell(const FEMSpace *fem, const Cell *cell,
                                               GradientCheckResult *result) {
  PetscFunctionBeginUser;
  const PetscInt nord = fem->nord;
  const PetscInt nNed = fem->numDofInCell;
  const PetscInt nH1  = fem->numH1DofInCell_Pnord;

  /* Fixed interior reference points (x,y,z >= 0, x+y+z <= 1). */
  const PetscReal pts[5][NUM_DIMENSIONS] = {
      {0.25, 0.25, 0.25}, {0.10, 0.20, 0.30}, {0.40, 0.10, 0.20},
      {0.20, 0.40, 0.10}, {0.15, 0.15, 0.50}};

  /* Exact gradient block (contiguous row-of-pointers, like the assembler). */
  PetscReal **G;
  PetscReal   laResidual = 0.0, condEst = 0.0;
  PetscCall(PetscCalloc1(nNed, &G));
  PetscCall(PetscCalloc1(nNed * nH1, &G[0]));
  for (PetscInt i = 1; i < nNed; i++) G[i] = G[i - 1] + nH1;
  PetscCall(buildExactDiscreteGradient(fem, cell, G, &laResidual, &condEst));

  /* H1 P_nord column entity classes, in DMPlex closure order volume->face->edge
   * ->vertex (= grid.H1dm_Pnord / shape3DHTet output). Counts from nord; empty
   * classes collapse to zero-width ranges. */
  const PetscInt nH1vol  = (nord - 1) * (nord - 2) * (nord - 3) / 6;
  const PetscInt nH1face = NUM_FACES_PER_CELL * (nord - 1) * (nord - 2) / 2;
  const PetscInt nH1edge = NUM_EDGES_PER_CELL * (nord - 1);
  const PetscInt volColEnd  = nH1vol;               /* [0, volColEnd) volume    */
  const PetscInt faceColEnd = nH1vol + nH1face;     /* [.., faceColEnd) face     */
  const PetscInt edgeColEnd = faceColEnd + nH1edge; /* [.., edgeColEnd) edge     */
  (void)edgeColEnd;                                 /* [edgeColEnd, nH1) vertex  */

  /* Nédélec row entity classes from fem offsets (empty classes have count 0). */
  const PetscInt edgeRow0 = fem->edgeDofOffset,   edgeRow1 = edgeRow0 + fem->numEdgeDof;
  const PetscInt faceRow0 = fem->faceDofOffset,   faceRow1 = faceRow0 + fem->numFaceDof;
  const PetscInt volRow0  = fem->volumeDofOffset, volRow1  = volRow0 + fem->numVolumeDof;

  /* Sparsity, per-entity row nnz, and forbidden (off-diagonal entity-block)
   * energy. Allowed de Rham couplings: edge-row -> {vertex,edge} H1; face-row ->
   * {vertex,edge,face} H1; volume-row -> any. The rest is "forbidden" and its
   * energy fraction is the entity-locality gate (gate 1b). */
  PetscInt  nnz = 0, maxEdge = 0, maxFace = 0, maxVol = 0;
  PetscReal forbidden2 = 0.0, total2 = 0.0;
  for (PetscInt i = 0; i < nNed; i++) {
    const PetscBool isEdge = (PetscBool)(i >= edgeRow0 && i < edgeRow1);
    const PetscBool isFace = (PetscBool)(i >= faceRow0 && i < faceRow1);
    const PetscBool isVol  = (PetscBool)(i >= volRow0  && i < volRow1);
    PetscInt rownz = 0;
    for (PetscInt k = 0; k < nH1; k++) {
      const PetscReal g = G[i][k];
      if (g == 0.0) continue;
      nnz++; rownz++;
      total2 += g * g;
      const PetscBool colVol  = (PetscBool)(k < volColEnd);
      const PetscBool colFace = (PetscBool)(k >= volColEnd && k < faceColEnd);
      if (isEdge && (colVol || colFace)) forbidden2 += g * g; /* edge-row off its edge */
      else if (isFace && colVol)         forbidden2 += g * g; /* face-row into volume  */
    }
    if (isEdge) maxEdge = PetscMax(maxEdge, rownz);
    if (isFace) maxFace = PetscMax(maxFace, rownz);
    if (isVol)  maxVol  = PetscMax(maxVol,  rownz);
  }

  /* Scratch and residual check. */
  PetscReal **Eref, **Cref, *Hval, **Hgrad;
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Eref));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Cref));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Hgrad));
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    PetscCall(PetscCalloc1(nNed, &Eref[d]));
    PetscCall(PetscCalloc1(nNed, &Cref[d]));
    PetscCall(PetscCalloc1(nH1,  &Hgrad[d]));
  }
  PetscCall(PetscCalloc1(nH1, &Hval));

  PetscReal res = 0.0;
  for (PetscInt p = 0; p < 5; p++) {
    PetscReal x[NUM_DIMENSIONS] = {pts[p][0], pts[p][1], pts[p][2]};
    PetscCall(shape3DETet(x, nord, &cell->orientation, Eref, Cref));
    PetscCall(shape3DHTet(x, nord, &cell->orientation, Hval, Hgrad));
    for (PetscInt k = 0; k < nH1; k++)
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
        PetscReal recon = 0.0;
        for (PetscInt i = 0; i < nNed; i++) recon += G[i][k] * Eref[d][i];
        res = PetscMax(res, PetscAbsReal(recon - Hgrad[d][k]));
      }
  }

  result->pointwiseResidual = res;
  result->laResidual        = laResidual;
  result->condEst           = condEst;
  result->crossEntityRatio  = (total2 > 0.0) ? PetscSqrtReal(forbidden2 / total2) : 0.0;
  result->nnzTotal          = nnz;
  result->maxEdgeRowNnz     = maxEdge;
  result->maxFaceRowNnz     = maxFace;
  result->maxVolRowNnz      = maxVol;

  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    PetscCall(PetscFree(Eref[d]));
    PetscCall(PetscFree(Cref[d]));
    PetscCall(PetscFree(Hgrad[d]));
  }
  PetscCall(PetscFree(Eref));
  PetscCall(PetscFree(Cref));
  PetscCall(PetscFree(Hgrad));
  PetscCall(PetscFree(Hval));
  PetscCall(PetscFree(G[0]));
  PetscCall(PetscFree(G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---- Per-order thin wrappers + dispatch tables -----------------------
 *
 * One ops table per supported order. All orders share
 * hierarchicalComputeBasisOrder / hierarchicalComputeCurlsOrder (Piola-pullback
 * wrappers around shape3DETet, parameterized by nord). The discrete gradient
 * for PCBDDC is the exact order-p operator built directly by
 * buildExactDiscreteGradient (no per-order dispatch). */
#define HIERARCHICAL_ORDER_OPS(N)                                                        \
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
  }

HIERARCHICAL_ORDER_OPS(1);
HIERARCHICAL_ORDER_OPS(2);
HIERARCHICAL_ORDER_OPS(3);
HIERARCHICAL_ORDER_OPS(4);
HIERARCHICAL_ORDER_OPS(5);
HIERARCHICAL_ORDER_OPS(6);

#undef HIERARCHICAL_ORDER_OPS
