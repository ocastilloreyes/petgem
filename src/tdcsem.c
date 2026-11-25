/*
 * Filename: tdcsem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-11-25
 *
 * Description:
 * Time-domain CSEM of fm.tdcsem (tdcsem.h).
 */

/* PETSc libraries */
#include <petscviewerhdf5.h>

/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "csem.h"
#include "fem.h"
#include "grid.h"
#include "io.h"
#include "receiver_interp.h"
#include "solver.h"
#include "tdcsem.h"

/** @brief simulation_type tag of fm.tdcsem products. */
#define PETGEM_SIM_TDCSEM "fm.tdcsem"

/** @brief Option values of -tdcsem_scheme, in TdScheme order. */
static const char *const tdSchemeNames[] = {"bdf2", "be"};

/**
 * @brief Copies the real part of a serial Vec into a new array.
 *
 * @param[in]  v    Serial Vec.
 * @param[out] n    Number of entries.
 * @param[out] out  Array of n entries; the caller frees it.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode copyRealArray(Vec v, PetscInt *n, PetscReal **out) {
  PetscFunctionBeginUser;

  const PetscScalar *a;

  PetscCall(VecGetSize(v, n));
  PetscCall(PetscMalloc1(*n, out));
  PetscCall(VecGetArrayRead(v, &a));
  for (PetscInt i = 0; i < *n; i++) (*out)[i] = PetscRealPart(a[i]);
  PetscCall(VecRestoreArrayRead(v, &a));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Mask of the interior nodal dofs.
 *
 * @param[in]  dm    DMPlex carrying the "Boundary" label.
 * @param[in]  grid  Grid with the nodal space H1dm.
 * @param[in]  G     Discrete gradient.
 * @param[out] mask  Vec on the column space of G: 0 on the boundary nodal dofs, 1 elsewhere.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode buildInteriorNodeMask(const DM dm, const Grid *grid, Mat G, Vec *mask) {
  PetscFunctionBeginUser;

  DMLabel         label = NULL;
  IS              bdIs  = NULL;
  PetscSection    gH1section;
  PetscInt        nbd = 0;
  const PetscInt *bd;

  PetscCall(MatCreateVecs(G, mask, NULL));
  PetscCall(VecSet(*mask, 1.0));
  PetscCall(DMGetGlobalSection(grid->H1dm, &gH1section));
  PetscCall(DMGetLabel(dm, "Boundary", &label));
  if (label) PetscCall(DMLabelGetStratumIS(label, 100, &bdIs));
  if (bdIs) {
    PetscCall(ISGetLocalSize(bdIs, &nbd));
    PetscCall(ISGetIndices(bdIs, &bd));
    for (PetscInt i = 0; i < nbd; i++) {
      PetscInt dof, off;
      PetscCall(PetscSectionGetDof(gH1section, bd[i], &dof));
      PetscCall(PetscSectionGetOffset(gH1section, bd[i], &off));
      if (dof > 0 && off >= 0) {
        for (PetscInt d = 0; d < dof; d++) PetscCall(VecSetValue(*mask, off + d, 0.0, INSERT_VALUES));
      }
    }
    PetscCall(ISRestoreIndices(bdIs, &bd));
    PetscCall(ISDestroy(&bdIs));
  }
  PetscCall(VecAssemblyBegin(*mask));
  PetscCall(VecAssemblyEnd(*mask));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Reads the -tdcsem_* options and sets the coefficients of the scheme.
 */
PetscErrorCode readTdParams(TdParams *td) {
  PetscFunctionBeginUser;

  PetscBool tEndSet = PETSC_FALSE, helpRequested = PETSC_FALSE;
  PetscInt  scheme = TD_SCHEME_BDF2;

  td->tEnd      = 0.0;
  td->dt0       = 0.0;
  td->atsdSteps = 50;
  td->atsdRtol  = 1.0e-2;

  PetscCall(PetscOptionsHasHelp(NULL, &helpRequested));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "fm.tdcsem: time-stepping options", "PETGEM");
  PetscCall(PetscOptionsReal("-tdcsem_t_end", "[REQUIRED] Final time (s)", "fm.tdcsem", td->tEnd, &td->tEnd, &tEndSet));
  PetscCall(PetscOptionsReal("-tdcsem_dt0", "Initial time step (s); default = smallest waveform sample spacing", "fm.tdcsem", td->dt0,
                             &td->dt0, NULL));
  PetscCall(PetscOptionsInt("-tdcsem_atsd_steps", "Steps at a fixed time step before a doubling trial", "fm.tdcsem", td->atsdSteps,
                            &td->atsdSteps, NULL));
  PetscCall(PetscOptionsReal("-tdcsem_atsd_rtol", "Relative tolerance of the doubling trial (0 = fixed time step)", "fm.tdcsem",
                             td->atsdRtol, &td->atsdRtol, NULL));
  PetscCall(
      PetscOptionsEList("-tdcsem_scheme", "Time integration scheme", "fm.tdcsem", tdSchemeNames, 2, tdSchemeNames[scheme], &scheme, NULL));
  PetscOptionsEnd();
  td->scheme = (TdScheme)scheme;
  if (td->scheme == TD_SCHEME_BDF2) {
    td->a[0] = 1.5, td->a[1] = -2.0, td->a[2] = 0.5;
  } else {
    td->a[0] = 1.0, td->a[1] = -1.0, td->a[2] = 0.0;
  }
  if (helpRequested) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCheck(tEndSet, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Missing required -tdcsem_t_end. Run with -help intro for usage.");
  PetscCheck(td->dt0 >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-tdcsem_dt0 must be >= 0 (got %g)", (double)td->dt0);
  PetscCheck(td->atsdSteps >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-tdcsem_atsd_steps must be >= 1 (got %" PetscInt_FMT ")",
             td->atsdSteps);
  PetscCheck(td->atsdRtol >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-tdcsem_atsd_rtol must be >= 0 (got %g)",
             (double)td->atsdRtol);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Loads the transmitters and the source waveform from the bundle.
 */
PetscErrorCode loadTdSettings(const PetgemParams *params, CsemSourceSet *sources, TdParams *td) {
  PetscFunctionBeginUser;

  const char    *names[] = {"position", "current", "length", "dipAngle", "azimuthAngle"};
  const PetscInt width[] = {3, 1, 1, 1, 1};
  PetscReal     *cols[5];
  PetscViewer    viewer;
  Vec            v;
  PetscInt       n, numSources = 0;

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, params->inputFile, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/sources"));
  for (PetscInt i = 0; i < 5; i++) {
    PetscCall(loadSerialVec(viewer, names[i], &v));
    PetscCall(copyRealArray(v, &n, &cols[i]));
    PetscCall(VecDestroy(&v));
    if (i == 0) numSources = n / width[0];
    PetscCheck(n == numSources * width[i], PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
               "%s: /sources/%s has %" PetscInt_FMT " entries, expected %" PetscInt_FMT, params->inputFile, names[i], n / width[i],
               numSources);
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  PetscCall(PetscViewerHDF5PushGroup(viewer, "/tdcsem"));
  PetscCall(loadSerialVec(viewer, "time", &v));
  PetscCall(copyRealArray(v, &td->numSamples, &td->time));
  PetscCall(VecDestroy(&v));
  PetscCall(loadSerialVec(viewer, "waveform", &v));
  PetscCall(copyRealArray(v, &n, &td->waveform));
  PetscCall(VecDestroy(&v));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCheck(numSources > 0, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "%s: /sources is empty", params->inputFile);
  sources->freq       = 0.0;
  sources->numSources = numSources;
  PetscCall(PetscMalloc1(numSources, &sources->sourceArray));
  for (PetscInt j = 0; j < numSources; j++) {
    CsemSource *s = &sources->sourceArray[j];
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) s->position[d] = cols[0][NUM_DIMENSIONS * j + d];
    s->current      = cols[1][j];
    s->length       = cols[2][j];
    s->dipAngle     = cols[3][j];
    s->azimuthAngle = cols[4][j];
  }
  for (PetscInt i = 0; i < 5; i++) PetscCall(PetscFree(cols[i]));

  PetscCheck(n == td->numSamples, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "%s: /tdcsem/waveform has %" PetscInt_FMT " samples, /tdcsem/time has %" PetscInt_FMT, params->inputFile, n, td->numSamples);
  PetscCheck(td->numSamples >= 2, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "%s: /tdcsem needs at least 2 waveform samples",
             params->inputFile);
  for (PetscInt i = 1; i < td->numSamples; i++) {
    PetscCheck(td->time[i] > td->time[i - 1], PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
               "%s: /tdcsem/time is not strictly increasing at sample %" PetscInt_FMT, params->inputFile, i);
  }
  PetscCheck(td->tEnd > td->time[0], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "-tdcsem_t_end = %g is not after the first waveform sample %g", (double)td->tEnd, (double)td->time[0]);

  if (td->dt0 == 0.0) {
    td->dt0 = td->time[1] - td->time[0];
    for (PetscInt i = 2; i < td->numSamples; i++) td->dt0 = PetscMin(td->dt0, td->time[i] - td->time[i - 1]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Frees the waveform arrays of the time-stepping options.
 */
PetscErrorCode destroyTdParams(TdParams *td) {
  PetscFunctionBeginUser;
  PetscCall(PetscFree(td->time));
  PetscCall(PetscFree(td->waveform));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the "CSEM source(s)" and "Time stepping" sections of the run report.
 */
PetscErrorCode logTdSetup(const CsemSourceSet *sources, const TdParams *td) {
  PetscFunctionBeginUser;

  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscCall(logSection(comm, sources->numSources == 1 ? "CSEM source" : "CSEM sources"));
  PetscCall(logKvInt(comm, "Number of sources", sources->numSources));
  for (PetscInt i = 0; i < sources->numSources; i++) {
    const CsemSource *s = &sources->sourceArray[i];
    PetscCall(PetscPrintf(comm, "   Source %" PetscInt_FMT ":\n", i + 1));
    PetscCall(PetscPrintf(comm, "     %-22s = (%s, %s, %s)\n", "Position (m)", formatReal(s->position[0]), formatReal(s->position[1]),
                          formatReal(s->position[2])));
    PetscCall(PetscPrintf(comm, "     %-22s = %s\n", "Current (A)", formatReal(s->current)));
    PetscCall(PetscPrintf(comm, "     %-22s = %s\n", "Length (m)", formatReal(s->length)));
    PetscCall(PetscPrintf(comm, "     %-22s = %s / %s\n", "Dip / azimuth (deg)", formatReal(s->dipAngle), formatReal(s->azimuthAngle)));
  }

  PetscCall(logSection(comm, "Time stepping"));
  PetscCall(logKvStr(comm, "Scheme", td->scheme == TD_SCHEME_BDF2 ? "BDF2" : "backward Euler"));
  PetscCall(logKvInt(comm, "Waveform samples", td->numSamples));
  PetscCall(logKvf(comm, "Waveform window (s)", "%s .. %s", formatReal(td->time[0]), formatReal(td->time[td->numSamples - 1])));
  PetscCall(logKvReal(comm, "Final time (s)", td->tEnd));
  PetscCall(logKvReal(comm, "Initial time step (s)", td->dt0));
  if (td->waveform[0] != 0.0) {
    PetscCall(logKvf(comm, "Initial field", "DC field x %s", formatReal(td->waveform[0])));
  } else {
    PetscCall(logKvStr(comm, "Initial field", "zero"));
  }
  if (td->atsdRtol > 0.0) {
    PetscCall(logKvf(comm, "Step doubling", "trial every %" PetscInt_FMT " steps, rtol %s", td->atsdSteps, formatReal(td->atsdRtol)));
  } else {
    PetscCall(logKvStr(comm, "Step doubling", "off"));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the "Solve" section of the run report.
 */
PetscErrorCode logTdSolve(MPI_Comm comm, const TdStats *stats, const TdCharge *charge, PetscReal dt, PetscReal t) {
  PetscFunctionBeginUser;
  PetscCall(logSection(comm, "Solve"));
  PetscCall(logKvStr(comm, "Solver", stats->solver));
  PetscCall(logKvInt(comm, "Time steps", stats->numSteps));
  PetscCall(logKvf(comm, "Operator setups", "%" PetscInt_FMT " (%" PetscInt_FMT " new)", stats->numSetups, stats->numCreated));
  PetscCall(logKvf(comm, "Doublings (acc / rej)", "%" PetscInt_FMT " / %" PetscInt_FMT, stats->numDoubled, stats->numRejected));
  PetscCall(logKvReal(comm, "Last time step (s)", dt));
  PetscCall(logKvReal(comm, "Last time (s)", t));
  PetscCall(logKvInt(comm, "KSP iterations", stats->its));
  if (charge->ksp) PetscCall(logKvInt(comm, "DC KSP iterations", charge->its));
  PetscCall(logKvf(comm, "Max charge residual", "%.3e", (double)charge->resMax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the header of the time-step table and starts the first row.
 */
PetscErrorCode beginTdMonitor(MPI_Comm comm, const TdStats *stats, TdMonitor *mon) {
  PetscFunctionBeginUser;
  PetscCall(logSection(comm, "Time steps"));
  PetscCall(PetscPrintf(comm, "   %6s %11s %11s %10s %8s %8s %10s %8s\n", "Steps", "t (s)", "dt (s)", "Doubling", "Result", "KSP its",
                        "Charge res", "Wall (s)"));
  mon->its = stats->its;
  PetscCall(PetscTime(&mon->tWall));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints one row of the time-step table and starts the next one.
 */
PetscErrorCode logTdStep(MPI_Comm comm, const TdStats *stats, PetscReal t, PetscReal dt, PetscReal ratio, const char *result, PetscReal res,
                         TdMonitor *mon) {
  PetscFunctionBeginUser;

  PetscLogDouble now;
  char           trial[16];

  if (ratio < 0.0)
    PetscCall(PetscStrncpy(trial, "-", sizeof(trial)));
  else
    PetscCall(PetscSNPrintf(trial, sizeof(trial), "%.3e", (double)ratio));
  PetscCall(PetscTime(&now));
  PetscCall(PetscPrintf(comm, "   %6" PetscInt_FMT " %11.4e %11.4e %10s %8s %8" PetscInt_FMT " %10.3e %8.1f\n", stats->numSteps, (double)t,
                        (double)dt, trial, result, stats->its - mon->its, (double)res, (double)(now - mon->tWall)));
  mon->its   = stats->its;
  mon->tWall = now;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Normalized source current s(t), by linear interpolation of the
 *        waveform samples; constant outside the sampled window.
 */
PetscReal evaluateWaveform(const TdParams *td, PetscReal t) {
  PetscInt lo = 0, hi = td->numSamples - 1;

  if (t <= td->time[lo]) return td->waveform[lo];
  if (t >= td->time[hi]) return td->waveform[hi];
  while (hi - lo > 1) {
    const PetscInt mid = (lo + hi) / 2;
    if (td->time[mid] <= t)
      lo = mid;
    else
      hi = mid;
  }
  return td->waveform[lo] + (td->waveform[hi] - td->waveform[lo]) * (t - td->time[lo]) / (td->time[hi] - td->time[lo]);
}

/**
 * @brief Assembles the curl-curl matrix K, the discrete gradient G and the
 *        conductivity-weighted mass matrix Ms on the pattern of K.
 */
PetscErrorCode assembleTdOperators(const DM dm, const Grid *grid, const Vec conductivity, Mat *K, Mat *Ms, Mat *G) {
  PetscFunctionBeginUser;

  const PetscInt nE = grid->fem.numDofInCell;
  Quadrature3D   quadrature3D;
  PetscReal    **Me, **Ke;

  PetscCall(assembleMaxwellOperator(dm, grid, conductivity, 0.0, K, NULL, G));
  PetscCall(MatDuplicate(*K, MAT_DO_NOT_COPY_VALUES, Ms));
  PetscCall(createQuadrature3D(grid->fem.order, &quadrature3D));
  PetscCall(createDenseRows(nE, nE, &Me));
  PetscCall(createDenseRows(nE, nE, &Ke));
  PetscCall(refillMassMatrix(dm, grid, conductivity, &quadrature3D, Me, Ke, *Ms));
  PetscCall(destroyDenseRows(&Me));
  PetscCall(destroyDenseRows(&Ke));
  PetscCall(destroyQuadrature3D(&quadrature3D));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the discrete gradient Gi restricted to the interior nodes and,
 *        for a step-off, the nodal operator L = Gi^T Ms Gi of the DC field.
 */
PetscErrorCode setupTdCharge(const DM dm, const Grid *grid, Mat Ms, Mat G, PetscBool solve, TdCharge *charge) {
  PetscFunctionBeginUser;

  Mat       MsAij;
  Vec       mask, boundary;
  PC        pc;
  PetscBool isIs;

  PetscCall(buildInteriorNodeMask(dm, grid, G, &mask));
  PetscCall(MatDuplicate(G, MAT_COPY_VALUES, &charge->Gi));
  PetscCall(MatDiagonalScale(charge->Gi, NULL, mask));
  PetscCall(MatCreateVecs(charge->Gi, &charge->phi, NULL));
  PetscCall(VecDuplicate(charge->phi, &charge->rhs));
  PetscCall(MatCreateVecs(Ms, NULL, &charge->h));
  charge->its    = 0;
  charge->resMax = 0.0;
  if (!solve) {
    PetscCall(VecDestroy(&mask));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)Ms, MATIS, &isIs));
  if (isIs) {
    PetscCall(MatConvert(Ms, MATAIJ, MAT_INITIAL_MATRIX, &MsAij));
  } else {
    PetscCall(PetscObjectReference((PetscObject)Ms));
    MsAij = Ms;
  }
  PetscCall(MatPtAP(MsAij, charge->Gi, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &charge->L));
  PetscCall(MatDestroy(&MsAij));

  PetscCall(VecDuplicate(mask, &boundary));
  PetscCall(VecSet(boundary, 1.0));
  PetscCall(VecAXPY(boundary, -1.0, mask));
  PetscCall(MatSetOption(charge->L, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatDiagonalSet(charge->L, boundary, ADD_VALUES));
  PetscCall(VecDestroy(&boundary));
  PetscCall(VecDestroy(&mask));
  PetscCall(MatSetOption(charge->L, MAT_SYMMETRIC, PETSC_TRUE));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)G), &charge->ksp));
  PetscCall(KSPSetOperators(charge->ksp, charge->L, charge->L));
  PetscCall(KSPSetType(charge->ksp, KSPPREONLY));
  PetscCall(KSPGetPC(charge->ksp, &pc));
  PetscCall(PCSetType(pc, PCCHOLESKY));
  PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
  PetscCall(KSPSetErrorIfNotConverged(charge->ksp, PETSC_TRUE));
  PetscCall(KSPSetOptionsPrefix(charge->ksp, "tdcsem_dc_"));
  PetscCall(KSPSetFromOptions(charge->ksp));
  PetscCall(KSPSetUp(charge->ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Destroys the objects of the discrete charge.
 */
PetscErrorCode destroyTdCharge(TdCharge *charge) {
  PetscFunctionBeginUser;
  PetscCall(MatDestroy(&charge->Gi));
  PetscCall(MatDestroy(&charge->L));
  PetscCall(KSPDestroy(&charge->ksp));
  PetscCall(VecDestroy(&charge->h));
  PetscCall(VecDestroy(&charge->rhs));
  PetscCall(VecDestroy(&charge->phi));
  PetscCall(PetscFree(charge->srcNorm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Charge residual max_i ||Gi^T (Ms E_i + s <N, p>_i)|| / ||Gi^T <N, p>_i||
 *        over the sources i, with <N, p> = -B0 / mu.
 */
PetscErrorCode computeTdChargeResidual(TdCharge *charge, Mat Ms, Mat B0, PetscReal s, Mat E, PetscReal *res) {
  PetscFunctionBeginUser;

  PetscInt numRhs;

  PetscCall(MatGetSize(E, NULL, &numRhs));
  if (!charge->srcNorm) {
    PetscCall(PetscMalloc1(numRhs, &charge->srcNorm));
    for (PetscInt i = 0; i < numRhs; i++) {
      Vec b;
      PetscCall(MatDenseGetColumnVecRead(B0, i, &b));
      PetscCall(MatMultTranspose(charge->Gi, b, charge->rhs));
      PetscCall(MatDenseRestoreColumnVecRead(B0, i, &b));
      PetscCall(VecNorm(charge->rhs, NORM_2, &charge->srcNorm[i]));
      charge->srcNorm[i] /= MU;
    }
  }
  *res = 0.0;
  for (PetscInt i = 0; i < numRhs; i++) {
    Vec       x, b;
    PetscReal r;
    PetscCall(MatDenseGetColumnVecRead(E, i, &x));
    PetscCall(MatMult(Ms, x, charge->h));
    PetscCall(MatDenseRestoreColumnVecRead(E, i, &x));
    if (s != 0.0) {
      PetscCall(MatDenseGetColumnVecRead(B0, i, &b));
      PetscCall(VecAXPY(charge->h, -s / MU, b));
      PetscCall(MatDenseRestoreColumnVecRead(B0, i, &b));
    }
    PetscCall(MatMultTranspose(charge->Gi, charge->h, charge->rhs));
    PetscCall(VecNorm(charge->rhs, NORM_2, &r));
    if (charge->srcNorm[i] > 0.0) *res = PetscMax(*res, r / charge->srcNorm[i]);
  }
  charge->resMax = PetscMax(charge->resMax, *res);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief DC field E = Gi phi with L phi = -s Gi^T <N, p> for every source,
 *        with <N, p> = -B0 / mu.
 */
PetscErrorCode computeTdDcField(TdCharge *charge, Mat Ms, Mat B0, PetscReal s, Mat E) {
  PetscFunctionBeginUser;

  PetscInt           numRhs, its;
  KSPConvergedReason reason;

  PetscCall(MatGetSize(E, NULL, &numRhs));
  for (PetscInt i = 0; i < numRhs; i++) {
    Vec x, b;
    PetscCall(MatDenseGetColumnVecRead(B0, i, &b));
    PetscCall(MatMultTranspose(charge->Gi, b, charge->rhs));
    PetscCall(MatDenseRestoreColumnVecRead(B0, i, &b));
    PetscCall(VecScale(charge->rhs, s / MU));
    PetscCall(KSPSolve(charge->ksp, charge->rhs, charge->phi));
    PetscCall(KSPGetIterationNumber(charge->ksp, &its));
    PetscCall(KSPGetConvergedReason(charge->ksp, &reason));
    PetscCheck(reason > 0, PetscObjectComm((PetscObject)Ms), PETSC_ERR_NOT_CONVERGED, "DC field failed: %s", KSPConvergedReasons[reason]);
    charge->its += its;
    PetscCall(MatDenseGetColumnVecWrite(E, i, &x));
    PetscCall(MatMult(charge->Gi, charge->phi, x));
    PetscCall(MatDenseRestoreColumnVecWrite(E, i, &x));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Forms A = K + (a0 mu/dt) Ms and sets up its KSP (symbolic and numeric
 *        setup).
 */
PetscErrorCode createTdStepper(Mat K, Mat Ms, Mat G, PetscInt order, IS primalVertices, PetscReal a0, PetscReal dt, TdStepper *stepper,
                               TdStats *stats) {
  PetscFunctionBeginUser;

  stepper->dt = dt;
  PetscCall(MatDuplicate(K, MAT_COPY_VALUES, &stepper->A));
  PetscCall(MatAXPY(stepper->A, a0 * MU / dt, Ms, SAME_NONZERO_PATTERN));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)K), &stepper->ksp));
  PetscCall(KSPSetOperators(stepper->ksp, stepper->A, stepper->A));
  PetscCall(setupBddcFromPetgemGradient(stepper->ksp, stepper->A, G, order, primalVertices));
  PetscCall(KSPSetFromOptions(stepper->ksp));
  PetscCall(KSPSetUp(stepper->ksp));
  stats->numSetups++;
  stats->numCreated++;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Refills A = K + (a0 mu/dt) Ms on its pattern and sets up its KSP again
 *        (numeric factorization only).
 */
PetscErrorCode refillTdStepper(Mat K, Mat Ms, PetscReal a0, PetscReal dt, TdStepper *stepper, TdStats *stats) {
  PetscFunctionBeginUser;

  stepper->dt = dt;
  PetscCall(MatCopy(K, stepper->A, SAME_NONZERO_PATTERN));
  PetscCall(MatAXPY(stepper->A, a0 * MU / dt, Ms, SAME_NONZERO_PATTERN));
  PetscCall(KSPSetUp(stepper->ksp));
  stats->numSetups++;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Destroys the operator and the KSP of a time step.
 */
PetscErrorCode destroyTdStepper(TdStepper *stepper) {
  PetscFunctionBeginUser;
  PetscCall(KSPDestroy(&stepper->ksp));
  PetscCall(MatDestroy(&stepper->A));
  stepper->dt = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief ||X||_K = (sum_i x_i^T K x_i)^(1/2) over the columns x_i of X.
 */
PetscErrorCode normTdCurl(Mat K, Mat X, PetscReal *nrm) {
  PetscFunctionBeginUser;

  PetscInt    numRhs;
  PetscScalar dot;
  PetscReal   sum = 0.0;
  Vec         w;

  PetscCall(MatGetSize(X, NULL, &numRhs));
  PetscCall(MatCreateVecs(K, NULL, &w));
  for (PetscInt i = 0; i < numRhs; i++) {
    Vec x;
    PetscCall(MatDenseGetColumnVecRead(X, i, &x));
    PetscCall(MatMult(K, x, w));
    PetscCall(VecDot(w, x, &dot));
    PetscCall(MatDenseRestoreColumnVecRead(X, i, &x));
    sum += PetscRealPart(dot);
  }
  PetscCall(VecDestroy(&w));
  *nrm = PetscSqrtReal(PetscMax(sum, 0.0));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief One time step from t to t + dt for every source.
 */
PetscErrorCode advanceTdStep(const TdStepper *stepper, Mat Ms, Mat B0, const TdParams *td, PetscReal t, Mat E0, Mat E1, Mat Enext, Mat R,
                             TdStats *stats) {
  PetscFunctionBeginUser;

  const PetscReal    dt = stepper->dt;
  const PetscReal   *a  = td->a;
  const PetscReal    ds = a[0] * evaluateWaveform(td, t + dt) + a[1] * evaluateWaveform(td, t) + a[2] * evaluateWaveform(td, t - dt);
  PetscInt           numRhs, its;
  KSPConvergedReason reason;

  PetscCall(MatCopy(E0, Enext, SAME_NONZERO_PATTERN));
  PetscCall(MatScale(Enext, a[1]));
  if (a[2] != 0.0) PetscCall(MatAXPY(Enext, a[2], E1, SAME_NONZERO_PATTERN));
  PetscCall(MatGetSize(R, NULL, &numRhs));
  for (PetscInt i = 0; i < numRhs; i++) {
    Vec x, r;
    PetscCall(MatDenseGetColumnVecRead(Enext, i, &x));
    PetscCall(MatDenseGetColumnVecWrite(R, i, &r));
    PetscCall(MatMult(Ms, x, r));
    PetscCall(MatDenseRestoreColumnVecWrite(R, i, &r));
    PetscCall(MatDenseRestoreColumnVecRead(Enext, i, &x));
  }
  PetscCall(MatScale(R, -MU / dt));
  if (ds != 0.0) PetscCall(MatAXPY(R, ds / dt, B0, SAME_NONZERO_PATTERN));

  PetscCall(KSPMatSolve(stepper->ksp, R, Enext));
  PetscCall(KSPGetIterationNumber(stepper->ksp, &its));
  PetscCall(KSPGetConvergedReason(stepper->ksp, &reason));
  PetscCheck(reason > 0, PetscObjectComm((PetscObject)Ms), PETSC_ERR_NOT_CONVERGED, "Time step at t = %g s (dt = %g s) failed: %s",
             (double)(t + dt), (double)dt, KSPConvergedReasons[reason]);
  stats->its += its;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the receiver interpolation and the empty time series.
 */
PetscErrorCode setupTdOutput(const CsemSourceSet sources, const DM dm, const Grid *grid, Vec receivers, TdOutput *out) {
  PetscFunctionBeginUser;

  const char *names[NUM_DIMENSIONS] = {"dBxdt", "dBydt", "dBzdt"};

  PetscCall(buildReceiverInterpolationMatrices(receivers, dm, grid, &out->Q));
  PetscCall(createReceiverFields(&out->Q, out->fields));
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) PetscCall(PetscObjectSetName((PetscObject)out->fields[RECV_HX + d], names[d]));
  PetscCall(VecGetLocalSize(out->fields[RECV_EX], &out->numLocal));
  out->numSources = sources.numSources;
  out->numTimes   = 0;
  out->capacity   = 0;
  out->times      = NULL;
  out->history    = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Appends E and dB/dt = -curl E at the receivers to the time series.
 */
PetscErrorCode recordTdSample(TdOutput *out, Mat E, PetscReal t) {
  PetscFunctionBeginUser;

  const PetscInt stride = out->numSources * NUM_RECEIVER_FIELDS * out->numLocal;

  if (out->numTimes == out->capacity) {
    out->capacity = PetscMax(2 * out->capacity, 256);
    PetscCall(PetscRealloc(out->capacity * sizeof(PetscReal), &out->times));
    PetscCall(PetscRealloc(out->capacity * stride * sizeof(PetscScalar), &out->history));
  }
  out->times[out->numTimes] = t;

  for (PetscInt i = 0; i < out->numSources; i++) {
    Vec x;
    PetscCall(MatDenseGetColumnVecRead(E, i, &x));
    PetscCall(interpolateReceiverFields(&out->Q, x, -1.0, out->fields));
    PetscCall(MatDenseRestoreColumnVecRead(E, i, &x));
    for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) {
      const PetscScalar *a;
      PetscScalar       *h = &out->history[out->numTimes * stride + (i * NUM_RECEIVER_FIELDS + comp) * out->numLocal];
      PetscCall(VecGetArrayRead(out->fields[comp], &a));
      for (PetscInt r = 0; r < out->numLocal; r++) h[r] = a[r];
      PetscCall(VecRestoreArrayRead(out->fields[comp], &a));
    }
  }
  out->numTimes++;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Writes the time series to a single HDF5 response file.
 */
PetscErrorCode writeTdOutput(const PetgemParams *params, const CsemSourceSet sources, const TdOutput *out) {
  PetscFunctionBeginUser;

  const PetscInt stride = out->numSources * NUM_RECEIVER_FIELDS * out->numLocal;
  const PetscInt nt     = out->numTimes;
  MPI_Comm       comm   = PetscObjectComm((PetscObject)out->fields[RECV_EX]);
  PetscMPIInt    rank;
  PetscViewer    viewer;
  Vec            v;
  PetscScalar   *a;
  char           outFileName[PETSC_MAX_PATH_LEN];
  char           groupPath[64];

  PetscCall(buildOutputPath(params, ".h5", outFileName, sizeof(outFileName)));
  PetscCall(PetscViewerHDF5Open(comm, outFileName, FILE_MODE_WRITE, &viewer));
  PetscCall(writeRunProvenance(viewer, params, PETGEM_SIM_TDCSEM));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "num_sources", PETSC_INT, &sources.numSources));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "num_times", PETSC_INT, &nt));

  /* /times, held by rank 0 */
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecCreateMPI(comm, rank == 0 ? nt : 0, nt, &v));
  PetscCall(PetscObjectSetName((PetscObject)v, "times"));
  PetscCall(VecGetArray(v, &a));
  if (rank == 0)
    for (PetscInt j = 0; j < nt; j++) a[j] = out->times[j];
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecView(v, viewer));
  PetscCall(VecDestroy(&v));

  for (PetscInt i = 0; i < sources.numSources; i++) {
    const CsemSource *s = &sources.sourceArray[i];

    /* /sources/src{k}/fields/{Ex..dBzdt}, one row per receiver */
    PetscCall(PetscSNPrintf(groupPath, sizeof(groupPath), "/sources/src%" PetscInt_FMT "/fields", i + 1));
    PetscCall(PetscViewerHDF5PushGroup(viewer, groupPath));
    for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) {
      const char *name;
      PetscCall(PetscObjectGetName((PetscObject)out->fields[comp], &name));
      PetscCall(VecCreate(comm, &v));
      PetscCall(VecSetSizes(v, out->numLocal * nt, PETSC_DECIDE));
      PetscCall(VecSetBlockSize(v, nt));
      PetscCall(VecSetType(v, VECSTANDARD));
      PetscCall(PetscObjectSetName((PetscObject)v, name));
      PetscCall(VecGetArray(v, &a));
      for (PetscInt r = 0; r < out->numLocal; r++) {
        for (PetscInt j = 0; j < nt; j++) a[r * nt + j] = out->history[j * stride + (i * NUM_RECEIVER_FIELDS + comp) * out->numLocal + r];
      }
      PetscCall(VecRestoreArray(v, &a));
      PetscCall(VecView(v, viewer));
      PetscCall(VecDestroy(&v));
    }
    PetscCall(PetscViewerHDF5PopGroup(viewer));

    /* /sources/src{k} attributes */
    PetscCall(PetscSNPrintf(groupPath, sizeof(groupPath), "/sources/src%" PetscInt_FMT, i + 1));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "x_pos", PETSC_REAL, &s->position[0]));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "y_pos", PETSC_REAL, &s->position[1]));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "z_pos", PETSC_REAL, &s->position[2]));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "current", PETSC_REAL, &s->current));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "length", PETSC_REAL, &s->length));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "dip_angle", PETSC_REAL, &s->dipAngle));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "azimuth_angle", PETSC_REAL, &s->azimuthAngle));
  }

  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Destroys the receiver interpolation and frees the time series.
 */
PetscErrorCode destroyTdOutput(TdOutput *out) {
  PetscFunctionBeginUser;
  PetscCall(destroyReceiverFields(out->fields));
  PetscCall(destroyReceiverInterpolationMatrices(&out->Q));
  PetscCall(PetscFree(out->times));
  PetscCall(PetscFree(out->history));
  PetscFunctionReturn(PETSC_SUCCESS);
}
