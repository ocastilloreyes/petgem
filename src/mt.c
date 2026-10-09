/*
 * Filename: mt.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Magnetotelluric (MT) layer: run options, box-face classification, 1D
 * conductivity profile of the lateral faces and the 1D boundary field H(z).
 */

/* PETSc libraries */
#include <petscdmplex.h>

/* PETGEM functions */
#include "constants.h"
#include "grid.h"
#include "mt.h"

/* Per-face record gathered for the 1D profile: zlo, zhi, sigma, h */
#define MT_FACE_RECORD 4

/**
 * @brief Reads the MT options -mt_1d_refine and -mt_1d_equation.
 *
 * @param[out] mt  MT options.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode readMtParams(MtParams *mt) {
  PetscFunctionBeginUser;

  const char *equations[] = {"paper", "h"};
  PetscInt    equation    = MT_1D_EQUATION_PAPER;

  mt->refine1D = 10;

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "fm.mt: MT options (optional)", "PETGEM");
  PetscCall(PetscOptionsInt("-mt_1d_refine", "1D element size = boundary 3D edge length / refine", "fm.mt", mt->refine1D, &mt->refine1D, NULL));
  PetscCall(PetscOptionsEList("-mt_1d_equation", "1D boundary equation: 'paper' (H'' + iωμσH = 0) or 'h' ((ρH')' + iωμH = 0)", "fm.mt",
                              equations, 2, equations[equation], &equation, NULL));
  PetscOptionsEnd();

  PetscCheck(mt->refine1D >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-mt_1d_refine must be >= 1 (got %" PetscInt_FMT ")", mt->refine1D);
  mt->equation1D = (Mt1DEquation)equation;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Assigns each boundary face to a box face Gamma_1..Gamma_6.
 *
 * @param[in]  dm     DMPlex mesh.
 * @param[in]  faces  Local boundary faces (getBoundaryFaces).
 * @param[out] tags   Box face of each entry of faces (caller frees).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode classifyMtBoxFaces(const DM dm, IS faces, MtBoxFace **tags) {
  PetscFunctionBeginUser;

  const MtBoxFace lowFace[NUM_DIMENSIONS]  = {MT_FACE_XMIN, MT_FACE_YMIN, MT_FACE_BOTTOM};
  const MtBoxFace highFace[NUM_DIMENSIONS] = {MT_FACE_XMAX, MT_FACE_YMAX, MT_FACE_TOP};
  PetscReal       lower[NUM_DIMENSIONS], upper[NUM_DIMENSIONS], extent = 0.0;
  const PetscInt *faceIdx;
  PetscInt        numFaces;

  PetscCall(DMGetBoundingBox(dm, lower, upper));
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) extent = PetscMax(extent, upper[d] - lower[d]);
  const PetscReal tolPlane = 1.0e-9 * extent;

  PetscCall(ISGetLocalSize(faces, &numFaces));
  PetscCall(ISGetIndices(faces, &faceIdx));
  PetscCall(PetscMalloc1(numFaces, tags));

  for (PetscInt i = 0; i < numFaces; i++) {
    PetscInt  cell, axis = 0;
    PetscReal vertices[NUM_VERTICES_PER_FACE][NUM_DIMENSIONS], normal[NUM_DIMENSIONS], area;

    PetscCall(computeBoundaryFaceGeometry(dm, faceIdx[i], &cell, vertices, normal, &area));
    for (PetscInt d = 1; d < NUM_DIMENSIONS; d++) {
      if (PetscAbsReal(normal[d]) > PetscAbsReal(normal[axis])) axis = d;
    }
    PetscCheck(PetscAbsReal(normal[axis]) > 1.0 - 1.0e-8, PETSC_COMM_SELF, PETSC_ERR_SUP,
               "classifyMtBoxFaces: boundary face %" PetscInt_FMT " has normal (%g, %g, %g); the MT domain must be an axis-aligned box",
               faceIdx[i], (double)normal[0], (double)normal[1], (double)normal[2]);

    const PetscReal plane = (normal[axis] > 0.0) ? upper[axis] : lower[axis];
    for (PetscInt v = 0; v < NUM_VERTICES_PER_FACE; v++) {
      PetscCheck(PetscAbsReal(vertices[v][axis] - plane) <= tolPlane, PETSC_COMM_SELF, PETSC_ERR_SUP,
                 "classifyMtBoxFaces: boundary face %" PetscInt_FMT " is off the bounding-box plane %c = %g; the MT domain must be an axis-aligned box",
                 faceIdx[i], "xyz"[axis], (double)plane);
    }
    (*tags)[i] = (normal[axis] > 0.0) ? highFace[axis] : lowFace[axis];
  }

  PetscCall(ISRestoreIndices(faces, &faceIdx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Compares two reals for qsort.
 */
static int compareReal(const void *a, const void *b) {
  const PetscReal x = *(const PetscReal *)a, y = *(const PetscReal *)b;
  return (x > y) - (x < y);
}

/**
 * @brief Builds the layered conductivity of the lateral faces Gamma_2..Gamma_5.
 *
 * @param[in]  dm            DMPlex mesh.
 * @param[in]  conductivity  Per-cell conductivity Vec.
 * @param[in]  faces         Local boundary faces (getBoundaryFaces).
 * @param[in]  tags          Box face of each entry of faces (classifyMtBoxFaces).
 * @param[out] profile       Layered conductivity, identical on every rank.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode buildMt1DProfile(const DM dm, const Vec conductivity, IS faces, const MtBoxFace *tags, Mt1DProfile *profile) {
  PetscFunctionBeginUser;

  MPI_Comm        comm = PetscObjectComm((PetscObject)dm);
  DM              dmConductivity;
  const PetscInt *faceIdx;
  PetscInt        numFaces, numLateral = 0;
  PetscReal      *local, *all, *breaks, *intervalSigma, *intervalH;
  PetscMPIInt     size, sendCount, *counts, *displs;
  PetscInt        totalRecords = 0;

  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(VecGetDM(conductivity, &dmConductivity));
  PetscCall(ISGetLocalSize(faces, &numFaces));
  PetscCall(ISGetIndices(faces, &faceIdx));

  /* Local records of the lateral faces */
  PetscCall(PetscMalloc1(MT_FACE_RECORD * numFaces, &local));
  for (PetscInt i = 0; i < numFaces; i++) {
    PetscInt  cell;
    PetscReal vertices[NUM_VERTICES_PER_FACE][NUM_DIMENSIONS], normal[NUM_DIMENSIONS], area, h = PETSC_MAX_REAL;
    Cell      owner;

    if (tags[i] == MT_FACE_TOP || tags[i] == MT_FACE_BOTTOM) continue;

    PetscCall(computeBoundaryFaceGeometry(dm, faceIdx[i], &cell, vertices, normal, &area));
    PetscCall(extractCellConductivity(dmConductivity, conductivity, cell, &owner));
    PetscCheck(PetscAbsReal(owner.conductivity[0] - owner.conductivity[1]) <= 1.0e-12 * PetscAbsReal(owner.conductivity[0]), PETSC_COMM_SELF, PETSC_ERR_SUP,
               "buildMt1DProfile: lateral cell %" PetscInt_FMT " has sigma_x = %g != sigma_y = %g", cell,
               (double)owner.conductivity[0], (double)owner.conductivity[1]);

    for (PetscInt a = 0; a < NUM_VERTICES_PER_FACE; a++) {
      const PetscInt b = (a + 1) % NUM_VERTICES_PER_FACE;
      PetscReal      len2 = 0.0;
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) len2 += (vertices[b][d] - vertices[a][d]) * (vertices[b][d] - vertices[a][d]);
      h = PetscMin(h, PetscSqrtReal(len2));
    }

    local[MT_FACE_RECORD * numLateral + 0] = PetscMin(vertices[0][2], PetscMin(vertices[1][2], vertices[2][2]));
    local[MT_FACE_RECORD * numLateral + 1] = PetscMax(vertices[0][2], PetscMax(vertices[1][2], vertices[2][2]));
    local[MT_FACE_RECORD * numLateral + 2] = owner.conductivity[0];
    local[MT_FACE_RECORD * numLateral + 3] = h;
    numLateral++;
  }
  PetscCall(ISRestoreIndices(faces, &faceIdx));

  /* Gather every lateral face on every rank */
  PetscCall(PetscMPIIntCast(MT_FACE_RECORD * numLateral, &sendCount));
  PetscCall(PetscMalloc2(size, &counts, size, &displs));
  PetscCallMPI(MPI_Allgather(&sendCount, 1, MPI_INT, counts, 1, MPI_INT, comm));
  for (PetscMPIInt r = 0; r < size; r++) {
    PetscCall(PetscMPIIntCast(totalRecords, &displs[r]));
    totalRecords += counts[r];
  }
  PetscCall(PetscMalloc1(totalRecords, &all));
  PetscCallMPI(MPI_Allgatherv(local, sendCount, MPIU_REAL, all, counts, displs, MPIU_REAL, comm));
  PetscCall(PetscFree2(counts, displs));
  PetscCall(PetscFree(local));

  const PetscInt numRecords = totalRecords / MT_FACE_RECORD;
  PetscCheck(numRecords > 0, comm, PETSC_ERR_ARG_WRONG, "buildMt1DProfile: no lateral boundary faces");

  /* Distinct heights of the lateral face vertices */
  PetscCall(PetscMalloc1(2 * numRecords, &breaks));
  for (PetscInt f = 0; f < numRecords; f++) {
    breaks[2 * f + 0] = all[MT_FACE_RECORD * f + 0];
    breaks[2 * f + 1] = all[MT_FACE_RECORD * f + 1];
  }
  qsort(breaks, (size_t)(2 * numRecords), sizeof(PetscReal), compareReal);
  const PetscReal tolZ = 1.0e-9 * (breaks[2 * numRecords - 1] - breaks[0]);
  PetscInt        numBreaks = 1;
  for (PetscInt i = 1; i < 2 * numRecords; i++) {
    if (breaks[i] - breaks[numBreaks - 1] > tolZ) breaks[numBreaks++] = breaks[i];
  }
  const PetscInt numIntervals = numBreaks - 1;
  PetscCheck(numIntervals >= 1, comm, PETSC_ERR_ARG_WRONG, "buildMt1DProfile: lateral faces have no vertical extent");

  /* Conductivity and edge length of each interval between consecutive heights */
  PetscCall(PetscMalloc2(numIntervals, &intervalSigma, numIntervals, &intervalH));
  for (PetscInt k = 0; k < numIntervals; k++) {
    intervalSigma[k] = -1.0;
    intervalH[k]     = PETSC_MAX_REAL;
  }
  for (PetscInt f = 0; f < numRecords; f++) {
    const PetscReal zlo = all[MT_FACE_RECORD * f + 0], zhi = all[MT_FACE_RECORD * f + 1];
    const PetscReal sigma = all[MT_FACE_RECORD * f + 2], h = all[MT_FACE_RECORD * f + 3];
    PetscInt        lo = 0, hi = numBreaks - 1;

    /* first break at or above zlo */
    while (lo < hi) {
      const PetscInt mid = (lo + hi) / 2;
      if (breaks[mid] < zlo - tolZ) lo = mid + 1;
      else hi = mid;
    }
    for (PetscInt k = lo; k < numIntervals && breaks[k] < zhi - tolZ; k++) {
      if (intervalSigma[k] < 0.0) {
        intervalSigma[k] = sigma;
      } else {
        PetscCheck(PetscAbsReal(intervalSigma[k] - sigma) <= 1.0e-10 * PetscAbsReal(sigma), comm, PETSC_ERR_SUP,
                   "buildMt1DProfile: lateral faces disagree on sigma at z in [%g, %g] (%g vs %g); the lateral boundaries must be 1D",
                   (double)breaks[k], (double)breaks[k + 1], (double)intervalSigma[k], (double)sigma);
      }
      intervalH[k] = PetscMin(intervalH[k], h);
    }
  }
  for (PetscInt k = 0; k < numIntervals; k++) {
    PetscCheck(intervalSigma[k] >= 0.0, comm, PETSC_ERR_ARG_WRONG, "buildMt1DProfile: no lateral face covers z in [%g, %g]",
               (double)breaks[k], (double)breaks[k + 1]);
  }

  /* Merge consecutive intervals with equal sigma into layers */
  PetscInt numLayers = 1;
  for (PetscInt k = 1; k < numIntervals; k++) {
    if (intervalSigma[k] != intervalSigma[k - 1]) numLayers++;
  }
  profile->numLayers = numLayers;
  PetscCall(PetscMalloc3(numLayers + 1, &profile->z, numLayers, &profile->sigma, numLayers, &profile->h));

  PetscInt layer = 0;
  profile->z[0]     = breaks[0];
  profile->sigma[0] = intervalSigma[0];
  profile->h[0]     = intervalH[0];
  for (PetscInt k = 1; k < numIntervals; k++) {
    if (intervalSigma[k] != intervalSigma[k - 1]) {
      profile->z[++layer]   = breaks[k];
      profile->sigma[layer] = intervalSigma[k];
      profile->h[layer]     = intervalH[k];
    } else {
      profile->h[layer] = PetscMin(profile->h[layer], intervalH[k]);
    }
  }
  profile->z[numLayers] = breaks[numIntervals];

  PetscCall(PetscFree2(intervalSigma, intervalH));
  PetscCall(PetscFree(breaks));
  PetscCall(PetscFree(all));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Solves the 1D boundary problem for H(z).
 *
 * @param[in]  profile  Layered conductivity.
 * @param[in]  mt       MT options (refine1D, equation1D).
 * @param[in]  omega    Angular frequency.
 * @param[out] field    Nodal H(z).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode solveMt1D(const Mt1DProfile *profile, const MtParams *mt, const PetscReal omega, Mt1DField *field) {
  PetscFunctionBeginUser;

  const PetscScalar iomegamu = PETSC_i * omega * MU;
  PetscInt         *numElements, numNodes = 1;
  PetscScalar      *lower, *diag, *upper, *rhs;

  /* 1D mesh: nodes at every interface, element size h/refine1D per layer */
  PetscCall(PetscMalloc1(profile->numLayers, &numElements));
  for (PetscInt l = 0; l < profile->numLayers; l++) {
    const PetscReal thickness = profile->z[l + 1] - profile->z[l];
    numElements[l] = PetscMax(1, (PetscInt)PetscCeilReal(thickness * mt->refine1D / profile->h[l]));
    numNodes += numElements[l];
  }

  field->numNodes = numNodes;
  PetscCall(PetscMalloc2(numNodes, &field->z, numNodes, &field->H));
  PetscCall(PetscCalloc4(numNodes, &lower, numNodes, &diag, numNodes, &upper, numNodes, &rhs));

  PetscInt node = 0;
  field->z[0] = profile->z[0];
  for (PetscInt l = 0; l < profile->numLayers; l++) {
    const PetscReal hElem = (profile->z[l + 1] - profile->z[l]) / numElements[l];
    const PetscReal sigma = profile->sigma[l];
    PetscScalar     kDiag, kOff, mDiag, mOff;

    /* Element matrix A_e = K_e - iωμ M_e */
    if (mt->equation1D == MT_1D_EQUATION_PAPER) {
      kDiag = 1.0 / hElem;
      mDiag = sigma * hElem / 3.0;
      mOff  = sigma * hElem / 6.0;
    } else {
      kDiag = 1.0 / (sigma * hElem);
      mDiag = hElem / 3.0;
      mOff  = hElem / 6.0;
    }
    kOff = -kDiag;
    const PetscScalar aDiag = kDiag - iomegamu * mDiag;
    const PetscScalar aOff  = kOff - iomegamu * mOff;

    for (PetscInt e = 0; e < numElements[l]; e++, node++) {
      diag[node]      += aDiag;
      diag[node + 1]  += aDiag;
      upper[node]     += aOff;
      lower[node + 1] += aOff;
      field->z[node + 1] = (e == numElements[l] - 1) ? profile->z[l + 1] : profile->z[l] + (e + 1) * hElem;
    }
  }
  PetscCall(PetscFree(numElements));

  /* Dirichlet: H(z_min) = 0, H(z_max) = 1 */
  diag[0]  = 1.0;
  upper[0] = 0.0;
  rhs[0]   = 0.0;
  diag[numNodes - 1]  = 1.0;
  lower[numNodes - 1] = 0.0;
  rhs[numNodes - 1]   = 1.0;

  /* Thomas algorithm */
  for (PetscInt i = 1; i < numNodes; i++) {
    const PetscScalar w = lower[i] / diag[i - 1];
    diag[i] -= w * upper[i - 1];
    rhs[i]  -= w * rhs[i - 1];
  }
  field->H[numNodes - 1] = rhs[numNodes - 1] / diag[numNodes - 1];
  for (PetscInt i = numNodes - 2; i >= 0; i--) {
    field->H[i] = (rhs[i] - upper[i] * field->H[i + 1]) / diag[i];
  }

  PetscCall(PetscFree4(lower, diag, upper, rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Evaluates H(z) by linear interpolation; z is clamped to [z_min, z_max].
 *
 * @param[in]  field  Nodal H(z).
 * @param[in]  z      Height.
 * @param[out] H      H(z).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode evalMt1DField(const Mt1DField *field, const PetscReal z, PetscScalar *H) {
  PetscFunctionBeginUser;

  const PetscInt n = field->numNodes;
  if (z <= field->z[0]) {
    *H = field->H[0];
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (z >= field->z[n - 1]) {
    *H = field->H[n - 1];
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* element [z[lo], z[lo+1]] containing z */
  PetscInt lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const PetscInt mid = (lo + hi) / 2;
    if (field->z[mid] <= z) lo = mid;
    else hi = mid;
  }
  const PetscReal t = (z - field->z[lo]) / (field->z[lo + 1] - field->z[lo]);
  *H = (1.0 - t) * field->H[lo] + t * field->H[lo + 1];

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Frees an Mt1DProfile.
 *
 * @param[in,out] profile  Profile to free.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode destroyMt1DProfile(Mt1DProfile *profile) {
  PetscFunctionBeginUser;
  PetscCall(PetscFree3(profile->z, profile->sigma, profile->h));
  profile->numLayers = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Frees an Mt1DField.
 *
 * @param[in,out] field  Field to free.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode destroyMt1DField(Mt1DField *field) {
  PetscFunctionBeginUser;
  PetscCall(PetscFree2(field->z, field->H));
  field->numNodes = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
