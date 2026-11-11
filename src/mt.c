/*
 * Filename: mt.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Magnetotelluric (MT) layer: run options, box-face classification, 1D
 * conductivity profile of the lateral faces, the 1D boundary field H(z), the
 * boundary right-hand side and the responses at the receivers.
 */

/* PETSc libraries */
#include <petscdmplex.h>
#include <petscviewerhdf5.h>

/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "fem.h"
#include "grid.h"
#include "io.h"
#include "mt.h"
#include "receiver_interp.h"

/* Per-face record gathered for the 1D profile: zlo, zhi, sigma, h */
#define MT_FACE_RECORD 4

/**
 * @brief Reads the MT options -mt_1d_refine and -mt_1d_equation.
 *
 * @param[out] mtParams  MT options.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode readMtParams(MtParams *mtParams) {
  PetscFunctionBeginUser;

  const char *equations[] = {"paper", "h"};
  PetscInt    equation    = MT_1D_EQUATION_PAPER;

  mtParams->refine1D = 10;

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "fm.mt: MT options (optional)", "PETGEM");
  PetscCall(PetscOptionsInt("-mt_1d_refine", "1D element size = boundary 3D edge length / refine", "fm.mt", mtParams->refine1D,
                            &mtParams->refine1D, NULL));
  PetscCall(PetscOptionsEList("-mt_1d_equation", "1D boundary equation: 'paper' (H'' + iωμσH = 0) or 'h' ((ρH')' + iωμH = 0)", "fm.mt",
                              equations, 2, equations[equation], &equation, NULL));
  PetscOptionsEnd();

  PetscCheck(mtParams->refine1D >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-mt_1d_refine must be >= 1 (got %" PetscInt_FMT ")",
             mtParams->refine1D);
  mtParams->equation1D = (Mt1DEquation)equation;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Reads the MT frequency from the bundle's /mt/freq dataset.
 *
 * @param[in]     params  Parameters (input file).
 * @param[in,out] mtParams      MT options; frequency is set.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode loadMtSettings(const PetgemParams *params, MtParams *mtParams) {
  PetscFunctionBeginUser;

  PetscViewer        viewer;
  Vec                freq;
  PetscInt           n;
  const PetscScalar *arr;

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, params->inputFile, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/mt"));
  PetscCall(VecCreate(PETSC_COMM_SELF, &freq));
  PetscCall(PetscObjectSetName((PetscObject)freq, "freq"));
  PetscCall(VecLoad(freq, viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(VecGetSize(freq, &n));
  PetscCheck(n == 1, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "loadMtSettings: /mt/freq has %" PetscInt_FMT " entries, expected 1 (bundle %s)", n, params->inputFile);
  PetscCall(VecGetArrayRead(freq, &arr));
  mtParams->frequency = PetscRealPart(arr[0]);
  PetscCall(VecRestoreArrayRead(freq, &arr));
  PetscCall(VecDestroy(&freq));

  PetscCheck(mtParams->frequency > 0.0, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "loadMtSettings: /mt/freq = %g is not positive (bundle %s)", (double)mtParams->frequency, params->inputFile);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Assigns each boundary face to a box face Gamma_1..Gamma_6.
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
                 "classifyMtBoxFaces: boundary face %" PetscInt_FMT
                 " is off the bounding-box plane %c = %g; the MT domain must be an axis-aligned box",
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
    PetscCheck(PetscAbsReal(owner.conductivity[0] - owner.conductivity[1]) <= 1.0e-12 * PetscAbsReal(owner.conductivity[0]),
               PETSC_COMM_SELF, PETSC_ERR_SUP, "buildMt1DProfile: lateral cell %" PetscInt_FMT " has sigma_x = %g != sigma_y = %g", cell,
               (double)owner.conductivity[0], (double)owner.conductivity[1]);

    for (PetscInt a = 0; a < NUM_VERTICES_PER_FACE; a++) {
      const PetscInt b    = (a + 1) % NUM_VERTICES_PER_FACE;
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
  for (PetscMPIInt rank = 0; rank < size; rank++) {
    PetscCall(PetscMPIIntCast(totalRecords, &displs[rank]));
    totalRecords += counts[rank];
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
  const PetscReal tolZ      = 1.0e-9 * (breaks[2 * numRecords - 1] - breaks[0]);
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
      if (breaks[mid] < zlo - tolZ)
        lo = mid + 1;
      else
        hi = mid;
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

  PetscInt layer    = 0;
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
 */
PetscErrorCode solveMt1D(const Mt1DProfile *profile, const MtParams *mtParams, Mt1DField *field) {
  PetscFunctionBeginUser;

  const PetscScalar iomegamu = iOmegaMu(mtParams->frequency);
  PetscInt         *numElements, numNodes = 1;
  PetscScalar      *lower, *diag, *upper, *rhs;

  /* 1D mesh: nodes at every interface, element size h/refine1D per layer */
  PetscCall(PetscMalloc1(profile->numLayers, &numElements));
  for (PetscInt l = 0; l < profile->numLayers; l++) {
    const PetscReal thickness = profile->z[l + 1] - profile->z[l];
    numElements[l]            = PetscMax(1, (PetscInt)PetscCeilReal(thickness * mtParams->refine1D / profile->h[l]));
    numNodes += numElements[l];
  }

  field->numNodes = numNodes;
  PetscCall(PetscMalloc2(numNodes, &field->z, numNodes, &field->H));
  PetscCall(PetscCalloc4(numNodes, &lower, numNodes, &diag, numNodes, &upper, numNodes, &rhs));

  PetscInt node = 0;
  field->z[0]   = profile->z[0];
  for (PetscInt l = 0; l < profile->numLayers; l++) {
    const PetscReal hElem = (profile->z[l + 1] - profile->z[l]) / numElements[l];
    const PetscReal sigma = profile->sigma[l];
    PetscScalar     kDiag, kOff, mDiag, mOff;

    /* Element matrix A_e = K_e - iωμ M_e */
    if (mtParams->equation1D == MT_1D_EQUATION_PAPER) {
      kDiag = 1.0 / hElem;
      mDiag = sigma * hElem / 3.0;
      mOff  = sigma * hElem / 6.0;
    } else {
      kDiag = 1.0 / (sigma * hElem);
      mDiag = hElem / 3.0;
      mOff  = hElem / 6.0;
    }
    kOff                    = -kDiag;
    const PetscScalar aDiag = kDiag - iomegamu * mDiag;
    const PetscScalar aOff  = kOff - iomegamu * mOff;

    for (PetscInt e = 0; e < numElements[l]; e++, node++) {
      diag[node] += aDiag;
      diag[node + 1] += aDiag;
      upper[node] += aOff;
      lower[node + 1] += aOff;
      field->z[node + 1] = (e == numElements[l] - 1) ? profile->z[l + 1] : profile->z[l] + (e + 1) * hElem;
    }
  }
  PetscCall(PetscFree(numElements));

  /* Dirichlet: H(z_min) = 0, H(z_max) = 1 */
  diag[0]             = 1.0;
  upper[0]            = 0.0;
  rhs[0]              = 0.0;
  diag[numNodes - 1]  = 1.0;
  lower[numNodes - 1] = 0.0;
  rhs[numNodes - 1]   = 1.0;

  /* Thomas algorithm */
  for (PetscInt i = 1; i < numNodes; i++) {
    const PetscScalar w = lower[i] / diag[i - 1];
    diag[i] -= w * upper[i - 1];
    rhs[i] -= w * rhs[i - 1];
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
    if (field->z[mid] <= z)
      lo = mid;
    else
      hi = mid;
  }
  const PetscReal t = (z - field->z[lo]) / (field->z[lo + 1] - field->z[lo]);
  *H                = (1.0 - t) * field->H[lo] + t * field->H[lo + 1];

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Assembles the MT right-hand side, Eq. (10), for both polarizations.
 *
 * This function integrates -iωμ N_j · (n × Ĥ) over every local boundary face
 * except Gamma_6 with the 2D triangle rule, evaluating the basis of the face's
 * support cell at the reference image of each quadrature point. Column 0 holds
 * the x-polarization, Ĥ = (0, H(z), 0); column 1 the y-polarization,
 * Ĥ = (H(z), 0, 0).
 *
 * @note The caller is responsible for destroying the returned matrix `B`.
 */
PetscErrorCode assembleMtBoundaryRhs(const PetscReal frequency, const DM dm, const Grid *grid, IS faces, const MtBoxFace *tags,
                                     const Mt1DField *field, Mat *B) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  Cell              cell;
  Quadrature2D      quadrature2D;
  PetscInt          numFaces, numDofIndices, *dofIndices;
  const PetscInt   *faceIdx;
  PetscReal       **Ni, XiEtaZeta[NUM_DIMENSIONS];
  PetscScalar      *closureRhs[MT_NUM_POLARIZATIONS];
  PetscSection      section;
  Vec               b[MT_NUM_POLARIZATIONS], bcol;
  MPI_Comm          comm        = PetscObjectComm((PetscObject)dm);
  const PetscScalar constFactor = -iOmegaMu(frequency);

  PetscCheck(grid->bc == PETGEM_BC_NATURAL, comm, PETSC_ERR_ARG_WRONGSTATE,
             "assembleMtBoundaryRHS: the grid must be set up with PETGEM_BC_NATURAL");

  /* One vector per polarization */
  for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) {
    PetscCall(createRhsVector(dm, &b[k]));
    PetscCall(VecZeroEntries(b[k]));
  }

  /* Matrix holding both right-hand sides */
  PetscCall(createRhsMatrix(b[0], MT_NUM_POLARIZATIONS, B));

  /* Get DM section */
  PetscCall(DMGetLocalSection(dm, &section));

  /* Compute 2D quadrature points */
  PetscCall(createQuadrature2D(grid->fem.order, &quadrature2D));

  /* Allocate memory */
  for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) {
    PetscCall(PetscCalloc1(grid->fem.numDofInCell, &closureRhs[k]));
  }
  PetscCall(createBasisValues(grid->fem.numDofInCell, &Ni));

  /* Boundary integral, face by face */
  PetscCall(ISGetLocalSize(faces, &numFaces));
  PetscCall(ISGetIndices(faces, &faceIdx));
  for (PetscInt f = 0; f < numFaces; f++) {
    PetscInt  cellId;
    PetscReal vertices[NUM_VERTICES_PER_FACE][NUM_DIMENSIONS], normal[NUM_DIMENSIONS], area;

    if (tags[f] == MT_FACE_BOTTOM) continue;

    /* Face geometry and support cell */
    PetscCall(computeBoundaryFaceGeometry(dm, faceIdx[f], &cellId, vertices, normal, &area));
    PetscCall(extractCellCoordinates(dm, cellId, &cell));

    for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) {
      for (PetscInt j = 0; j < grid->fem.numDofInCell; j++) closureRhs[k][j] = 0.0;
    }

    for (PetscInt q = 0; q < quadrature2D.numPoints; q++) {
      const PetscReal s = quadrature2D.points[q][0], t = quadrature2D.points[q][1];
      PetscReal       point[NUM_DIMENSIONS];
      PetscScalar     H, nxH[MT_NUM_POLARIZATIONS][NUM_DIMENSIONS];

      /* Physical quadrature point and its image in the reference cell */
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
        point[d] = vertices[0][d] + s * (vertices[1][d] - vertices[0][d]) + t * (vertices[2][d] - vertices[0][d]);
      }
      PetscCall(tetrahedronXyzToReference(cell.coordinates, point, XiEtaZeta));
      PetscCall(evaluateNedelecBasis(&grid->fem, &cell, XiEtaZeta, Ni, NULL));

      /* n x Ĥ: x-polarization Ĥ = (0, H, 0), y-polarization Ĥ = (H, 0, 0) */
      PetscCall(evalMt1DField(field, point[2], &H));
      nxH[0][0] = -normal[2] * H;
      nxH[0][1] = 0.0;
      nxH[0][2] = normal[0] * H;
      nxH[1][0] = 0.0;
      nxH[1][1] = normal[2] * H;
      nxH[1][2] = -normal[1] * H;

      const PetscReal wdet = quadrature2D.weights[q] * 2.0 * area;
      for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) {
        for (PetscInt j = 0; j < grid->fem.numDofInCell; j++) {
          PetscScalar dot = 0.0;
          for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
            dot += (PetscScalar)Ni[d][j] * nxH[k][d];
          }
          closureRhs[k][j] += wdet * constFactor * dot;
        }
      }
    }

    /* Add face contribution through the support cell closure */
    PetscCall(DMPlexGetClosureIndices(dm, section, section, cellId, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
    for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) {
      PetscCall(VecSetValuesLocal(b[k], numDofIndices, dofIndices, closureRhs[k], ADD_VALUES));
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, cellId, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
  }
  PetscCall(ISRestoreIndices(faces, &faceIdx));

  /* Global assembly and copy into B */
  for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) {
    PetscCall(VecAssemblyBegin(b[k]));
    PetscCall(VecAssemblyEnd(b[k]));
    PetscCall(MatDenseGetColumnVecWrite(*B, k, &bcol));
    PetscCall(VecCopy(b[k], bcol));
    PetscCall(MatDenseRestoreColumnVecWrite(*B, k, &bcol));
    PetscCall(VecDestroy(&b[k]));
  }

  /* Free memory */
  PetscCall(destroyQuadrature2D(&quadrature2D));
  PetscCall(destroyBasisValues(&Ni));
  for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) {
    PetscCall(PetscFree(closureRhs[k]));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the "MT" section of the run report.
 *
 * @param[in] comm      Communicator.
 * @param[in] mtParams        MT options.
 * @param[in] faces     Local boundary faces.
 * @param[in] profile   Layered conductivity of the lateral faces.
 * @param[in] field     Nodal H(z).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode logMtSetup(MPI_Comm comm, const MtParams *mtParams, IS faces, const Mt1DProfile *profile, const Mt1DField *field) {
  PetscFunctionBeginUser;

  PetscInt numFaces, numFacesGlobal;

  PetscCall(ISGetLocalSize(faces, &numFaces));
  PetscCallMPI(MPI_Allreduce(&numFaces, &numFacesGlobal, 1, MPIU_INT, MPI_SUM, comm));

  PetscCall(logSection(comm, "MT"));
  PetscCall(logKvReal(comm, "Frequency (Hz)", mtParams->frequency));
  PetscCall(logKvStr(comm, "Polarizations", "x, y"));
  PetscCall(logKvInt(comm, "Boundary faces", numFacesGlobal));
  PetscCall(logKvStr(comm, "1D equation", mtParams->equation1D == MT_1D_EQUATION_PAPER ? "H'' + iωμσH = 0" : "(ρH')' + iωμH = 0"));
  PetscCall(logKvInt(comm, "1D refinement", mtParams->refine1D));
  PetscCall(logKvInt(comm, "1D nodes", field->numNodes));
  PetscCall(logKvInt(comm, "1D layers", profile->numLayers));
  for (PetscInt l = profile->numLayers - 1; l >= 0; l--) {
    char key[32];
    PetscCall(PetscSNPrintf(key, sizeof(key), "  Layer %" PetscInt_FMT, profile->numLayers - l));
    PetscCall(
        logKvf(comm, key, "z = [%g, %g] m, sigma = %g S/m", (double)profile->z[l], (double)profile->z[l + 1], (double)profile->sigma[l]));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode createNamedLike(Vec tmpl, const char *name, Vec *v) {
  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(tmpl, v));
  PetscCall(PetscObjectSetName((PetscObject)*v, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode viewGroup(PetscViewer viewer, const char *group, Vec v[], PetscInt n) {
  PetscFunctionBeginUser;
  PetscCall(PetscViewerHDF5PushGroup(viewer, group));
  for (PetscInt comp = 0; comp < n; comp++) PetscCall(VecView(v[comp], viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Per receiver: Z = [E1 E2][H1 H2]^-1, T = [Hz1 Hz2][H1 H2]^-1 (Eq. A.3), rho (Eq. A.4), phi in degrees (Eq. A.5). */
static PetscErrorCode computeImpedance(Vec fields[MT_NUM_POLARIZATIONS][NUM_RECEIVER_FIELDS], PetscReal omega, Vec Z[4], Vec rho[4],
                                       Vec phi[4], Vec T[2]) {
  PetscFunctionBeginUser;
  const PetscScalar *f[MT_NUM_POLARIZATIONS][NUM_RECEIVER_FIELDS];
  PetscScalar       *z[4], *r[4], *p[4], *t[2];
  PetscInt           n;

  PetscCall(VecGetLocalSize(fields[0][0], &n));
  for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) {
    for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) PetscCall(VecGetArrayRead(fields[k][comp], &f[k][comp]));
  }
  for (PetscInt comp = 0; comp < 4; comp++) {
    PetscCall(VecGetArray(Z[comp], &z[comp]));
    PetscCall(VecGetArray(rho[comp], &r[comp]));
    PetscCall(VecGetArray(phi[comp], &p[comp]));
  }
  for (PetscInt comp = 0; comp < 2; comp++) PetscCall(VecGetArray(T[comp], &t[comp]));

  for (PetscInt i = 0; i < n; i++) {
    const PetscScalar E1x = f[0][RECV_EX][i], E1y = f[0][RECV_EY][i];
    const PetscScalar H1x = f[0][RECV_HX][i], H1y = f[0][RECV_HY][i], H1z = f[0][RECV_HZ][i];
    const PetscScalar E2x = f[1][RECV_EX][i], E2y = f[1][RECV_EY][i];
    const PetscScalar H2x = f[1][RECV_HX][i], H2y = f[1][RECV_HY][i], H2z = f[1][RECV_HZ][i];
    /* [H1 H2]^-1 = [H2y -H2x; -H1y H1x] / det */
    const PetscScalar det = H1x * H2y - H2x * H1y;

    z[0][i] = (E1x * H2y - E2x * H1y) / det;
    z[1][i] = (E2x * H1x - E1x * H2x) / det;
    z[2][i] = (E1y * H2y - E2y * H1y) / det;
    z[3][i] = (E2y * H1x - E1y * H2x) / det;
    t[0][i] = (H1z * H2y - H2z * H1y) / det;
    t[1][i] = (H2z * H1x - H1z * H2x) / det;
    for (PetscInt comp = 0; comp < 4; comp++) {
      r[comp][i] = PetscSqr(PetscAbsScalar(z[comp][i])) / (omega * MU);
      p[comp][i] = PetscAtan2Real(PetscImaginaryPart(z[comp][i]), PetscRealPart(z[comp][i])) * 180.0 / PETSC_PI;
    }
  }

  for (PetscInt comp = 0; comp < 2; comp++) PetscCall(VecRestoreArray(T[comp], &t[comp]));
  for (PetscInt comp = 0; comp < 4; comp++) {
    PetscCall(VecRestoreArray(phi[comp], &p[comp]));
    PetscCall(VecRestoreArray(rho[comp], &r[comp]));
    PetscCall(VecRestoreArray(Z[comp], &z[comp]));
  }
  for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) {
    for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) PetscCall(VecRestoreArrayRead(fields[k][comp], &f[k][comp]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the MT responses at the receivers and writes them to HDF5.
 *
 * This function interpolates E and H = curl(E)/(iωμ) of both polarizations at
 * the receivers with the receiver-interpolation operators, solves the 2x2
 * systems of Eq. (A.3) per receiver for the impedance and the tipper, derives
 * the apparent resistivity and phase, and writes everything to a single HDF5
 * file through PETSc's HDF5 viewer.
 */
PetscErrorCode computeMtResponses(const PetgemParams *params, const MtParams *mtParams, const DM dm, const Grid *grid, Vec receivers,
                                  const Mat X) {
  PetscFunctionBeginUser;

  const char                   *polGroups[MT_NUM_POLARIZATIONS] = {"/polarizations/x/fields", "/polarizations/y/fields"};
  const char                   *tensorNames[4]                  = {"xx", "xy", "yx", "yy"};
  const char                   *tipperNames[2]                  = {"x", "y"};
  const PetscReal               omega                           = angularFrequency(mtParams->frequency);
  const PetscScalar             constFactor                     = iOmegaMu(mtParams->frequency);
  ReceiverInterpolationMatrices Q;
  Vec                           fields[MT_NUM_POLARIZATIONS][NUM_RECEIVER_FIELDS], Z[4], rho[4], phi[4], T[2];
  PetscViewer                   viewer;
  PetscInt                      numReceivers;
  char                          outFileName[PETSC_MAX_PATH_LEN];
  MPI_Comm                      comm = PetscObjectComm((PetscObject)dm);

  /* E and H at the receivers for each polarization */
  PetscCall(buildReceiverInterpolationMatrices(receivers, dm, grid, &Q));
  for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) {
    Vec x;
    PetscCall(createReceiverFields(&Q, fields[k]));
    PetscCall(MatDenseGetColumnVecRead(X, k, &x));
    PetscCall(interpolateReceiverFields(&Q, x, constFactor, fields[k]));
    PetscCall(MatDenseRestoreColumnVecRead(X, k, &x));
  }
  PetscCall(destroyReceiverInterpolationMatrices(&Q));

  for (PetscInt comp = 0; comp < 4; comp++) {
    PetscCall(createNamedLike(fields[0][0], tensorNames[comp], &Z[comp]));
    PetscCall(createNamedLike(fields[0][0], tensorNames[comp], &rho[comp]));
    PetscCall(createNamedLike(fields[0][0], tensorNames[comp], &phi[comp]));
  }
  for (PetscInt comp = 0; comp < 2; comp++) PetscCall(createNamedLike(fields[0][0], tipperNames[comp], &T[comp]));
  PetscCall(computeImpedance(fields, omega, Z, rho, phi, T));

  /* Single output file: {output_dir}/{output_filename}.h5 */
  PetscCall(buildOutputPath(params, ".h5", outFileName, sizeof(outFileName)));
  PetscCall(PetscViewerHDF5Open(comm, outFileName, FILE_MODE_WRITE, &viewer));
  PetscCall(writeRunProvenance(viewer, params, PETGEM_SIM_MT));
  PetscCall(VecGetSize(fields[0][0], &numReceivers));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "frequency", PETSC_REAL, &mtParams->frequency));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "num_receivers", PETSC_INT, &numReceivers));
  for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) PetscCall(viewGroup(viewer, polGroups[k], fields[k], NUM_RECEIVER_FIELDS));
  PetscCall(viewGroup(viewer, "/impedance", Z, 4));
  PetscCall(viewGroup(viewer, "/apparent_resistivity", rho, 4));
  PetscCall(viewGroup(viewer, "/phase", phi, 4));
  PetscCall(viewGroup(viewer, "/tipper", T, 2));
  PetscCall(PetscViewerDestroy(&viewer));

  for (PetscInt k = 0; k < MT_NUM_POLARIZATIONS; k++) PetscCall(destroyReceiverFields(fields[k]));
  for (PetscInt comp = 0; comp < 4; comp++) {
    PetscCall(VecDestroy(&Z[comp]));
    PetscCall(VecDestroy(&rho[comp]));
    PetscCall(VecDestroy(&phi[comp]));
  }
  for (PetscInt comp = 0; comp < 2; comp++) PetscCall(VecDestroy(&T[comp]));
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
