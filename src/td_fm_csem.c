/*
 * Filename: td_fm_csem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-12
 *
 * Description:
 * Time-domain forward CSEM kernel (runTdCsemForward), linked into fm.tdcsem
 * and petgem.
 */

/*
 * Notes:
 * BDF2 (second-order backward differentiation formula, default) or backward
 * Euler discretization of curl curl E + mu sigma dE/dt = -mu dJs/dt with
 * adaptive time step doubling (Cai et al., 2017, Eqs. 5-14). With D = Ms E + s <N, p> and coefficients
 * a = (3/2, -2, 1/2) for BDF2, (1, -1, 0) for backward Euler:
 *   K E(t) + mu/dt (a0 D(t) + a1 D(t-dt) + a2 D(t-2dt)) = 0
 * A doubling to 2dt is accepted when ||E_2dt - E_dt|| <= rtol ||E(t+2dt) - E(t)||;
 * the BDF2 trial step uses E(t) and E(t-2dt).
 * After every solve, E += G phi with (G^T Ms G) phi = -G^T (Ms E + s(t) <N, p>),
 * phi = 0 on the boundary nodes.
 * Zero initial field, PEC boundary.
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
#include "kernels.h"
#include "receiver_interp.h"
#include "runtime.h"
#include "solver.h"

/** @brief simulation_type tag of fm.tdcsem products. */
#define PETGEM_SIM_TDCSEM "fm.tdcsem"

/** @brief Time integration scheme. */
typedef enum { TD_SCHEME_BDF2, TD_SCHEME_BE } TdScheme;

static const char *const tdSchemeNames[] = {"bdf2", "be"};

/**
 * @brief Source waveform and time-stepping options.
 */
typedef struct {
  PetscInt   numSamples; /**< Waveform samples, from the bundle's /tdcsem group. */
  PetscReal *time;       /**< Sample times (s), strictly increasing. */
  PetscReal *waveform;   /**< Normalized source current s(t) at each sample. */
  PetscReal  tEnd;       /**< Final time (s). */
  PetscReal  dt0;        /**< Initial time step (s); 0 = smallest sample spacing. */
  PetscInt   atsdSteps;  /**< Steps at a fixed time step before a doubling trial. */
  PetscReal  atsdRtol;   /**< Doubling tolerance; 0 = fixed time step. */
  TdScheme   scheme;     /**< Time integration scheme. */
  PetscReal  a[3];       /**< Coefficients of D(t), D(t-dt), D(t-2dt). */
} TdParams;

/**
 * @brief Operator A = K + (a0 mu/dt) Ms of one time step and its KSP.
 */
typedef struct {
  PetscReal dt;  /**< Time step (s). */
  Mat       A;   /**< Backward Euler operator. */
  KSP       ksp; /**< Solver bound to A, set up once. */
} TdStepper;

/**
 * @brief Charge correction: nodal operator L = Gi^T Ms Gi and its KSP.
 */
typedef struct {
  Mat      Gi;  /**< Discrete gradient restricted to the interior nodal columns. */
  Mat      L;   /**< Gi^T Ms Gi, identity on the boundary nodes. */
  KSP      ksp; /**< CG + GAMG on L, set up once. */
  Vec      h;   /**< Work Vec on the H(curl) space. */
  Vec      rhs; /**< Nodal right-hand side. */
  Vec      phi; /**< Nodal correction potential. */
  PetscInt its; /**< KSP iterations over all corrections. */
} TdCharge;

/**
 * @brief Time-stepping counters for the run report.
 */
typedef struct {
  PetscInt numSteps;    /**< Accepted time steps. */
  PetscInt numSetups;   /**< Operators assembled and set up. */
  PetscInt numDoubled;  /**< Accepted doubling trials. */
  PetscInt numRejected; /**< Rejected doubling trials. */
  PetscInt its;         /**< KSP iterations over all solves. */
  char     solver[32];  /**< PC type, or factorization package. */
} TdStats;

/**
 * @brief Counters at the start of the current row of the time-step table.
 */
typedef struct {
  PetscInt       its;       /**< TdStats.its at the row start. */
  PetscInt       chargeIts; /**< TdCharge.its at the row start. */
  PetscLogDouble tWall;     /**< Wall-clock time at the row start. */
} TdMonitor;

/**
 * @brief Receiver interpolation and the sampled time series.
 */
typedef struct {
  ReceiverInterpolationMatrices Q;                           /**< Receiver interpolation operators. */
  Vec                           fields[NUM_RECEIVER_FIELDS]; /**< E and dB/dt at the receivers. */
  PetscInt                      numSources;                  /**< Transmitters. */
  PetscInt                      numLocal;                    /**< Receivers owned by this rank. */
  PetscInt                      numTimes;                    /**< Samples recorded. */
  PetscInt                      capacity;                    /**< Samples allocated. */
  PetscReal                    *times;                       /**< Sample times (s). */
  PetscScalar                  *history;                     /**< Owned values, [time][source][field][receiver]. */
} TdOutput;

static char tdHelp[] = "================================================================\n\
 PETGEM  fm.tdcsem  --  time-domain CSEM forward modeling kernel\n\
================================================================\n\
\n\
 Computes the 3-D transient controlled-source EM response on a\n\
 tetrahedral mesh using high-order (1..6) Nedelec finite elements\n\
 and BDF2 or backward Euler time stepping with adaptive step\n\
 doubling.\n\
\n\
QUICK START\n\
  mpirun -n <np> ./fm.tdcsem -options_file params.txt\n\
\n\
REQUIRED\n\
  -input_filename <file.h5>  Unified PETGEM bundle (mesh, model,\n\
                             sources, waveform, receivers).\n\
  -output_dir <dir>          Output directory (created if absent).\n\
  -output_filename <stem>    Output stem; writes <dir>/<stem>.h5.\n\
  -tdcsem_t_end <s>          Final time.\n\
\n\
OPTIONAL  (defaults in brackets)\n\
  -order <1..6>              Override bundle basis order.\n\
  -tdcsem_dt0 <s>            Initial time step  [smallest waveform\n\
                             sample spacing]\n\
  -tdcsem_atsd_steps <n>     Steps before a doubling trial   [100]\n\
  -tdcsem_atsd_rtol <r>      Doubling tolerance, 0 = fixed  [1e-2]\n\
  -tdcsem_scheme <s>         bdf2 | be                    [bdf2]\n\
\n\
EXAMPLES\n\
  mpirun -n 96 ./fm.tdcsem -options_file params.txt\n\
\n\
UNIFIED BINARY\n\
  mpirun -n 96 ./petgem fm.tdcsem -options_file params.txt\n\
\n" PETGEM_HELP_FOOTER;

/* Reads the -tdcsem_* options. */
static PetscErrorCode readTdParams(TdParams *td) {
  PetscFunctionBeginUser;

  PetscBool tEndSet = PETSC_FALSE, helpRequested = PETSC_FALSE;
  PetscInt  scheme = TD_SCHEME_BDF2;

  td->tEnd      = 0.0;
  td->dt0       = 0.0;
  td->atsdSteps = 100;
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

/* Copies the real part of a serial Vec into a new array. */
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
 * @brief Loads the transmitters (/sources, frequency unused) and the waveform
 *        (/tdcsem/time, /tdcsem/waveform) from the bundle.
 */
static PetscErrorCode loadTdSettings(const PetgemParams *params, CsemSourceSet *sources, TdParams *td) {
  PetscFunctionBeginUser;

  const char    *names[] = {"position", "current", "length", "dipAngle", "azimuthAngle"};
  const PetscInt width[] = {3, 1, 1, 1, 1};
  PetscReal     *cols[5];
  PetscViewer    viewer;
  Vec            v;
  PetscInt       n, numSources = 0;

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, params->inputFile, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/sources"));
  for (PetscInt k = 0; k < 5; k++) {
    PetscCall(loadSerialVec(viewer, names[k], &v));
    PetscCall(copyRealArray(v, &n, &cols[k]));
    PetscCall(VecDestroy(&v));
    if (k == 0) numSources = n / width[0];
    PetscCheck(n == numSources * width[k], PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
               "%s: /sources/%s has %" PetscInt_FMT " entries, expected %" PetscInt_FMT, params->inputFile, names[k], n / width[k],
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
  for (PetscInt i = 0; i < numSources; i++) {
    CsemSource *s = &sources->sourceArray[i];
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) s->position[d] = cols[0][3 * i + d];
    s->current      = cols[1][i];
    s->length       = cols[2][i];
    s->dipAngle     = cols[3][i];
    s->azimuthAngle = cols[4][i];
  }
  for (PetscInt k = 0; k < 5; k++) PetscCall(PetscFree(cols[k]));

  PetscCheck(n == td->numSamples, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "%s: /tdcsem/waveform has %" PetscInt_FMT " samples, /tdcsem/time has %" PetscInt_FMT, params->inputFile, n, td->numSamples);
  PetscCheck(td->numSamples >= 2, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "%s: /tdcsem needs at least 2 waveform samples",
             params->inputFile);
  for (PetscInt i = 1; i < td->numSamples; i++) {
    PetscCheck(td->time[i] > td->time[i - 1], PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
               "%s: /tdcsem/time is not strictly increasing at sample %" PetscInt_FMT, params->inputFile, i);
  }
  PetscCheck(td->waveform[0] == 0.0, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "%s: waveform starts at %g; only a zero initial current is supported (no step-off)", params->inputFile,
             (double)td->waveform[0]);
  PetscCheck(td->tEnd > td->time[0], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "-tdcsem_t_end = %g is not after the first waveform sample %g", (double)td->tEnd, (double)td->time[0]);

  if (td->dt0 == 0.0) {
    td->dt0 = td->time[1] - td->time[0];
    for (PetscInt i = 2; i < td->numSamples; i++) td->dt0 = PetscMin(td->dt0, td->time[i] - td->time[i - 1]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyTdParams(TdParams *td) {
  PetscFunctionBeginUser;
  PetscCall(PetscFree(td->time));
  PetscCall(PetscFree(td->waveform));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Prints the "CSEM source(s)" and "Time stepping" sections of the run report. */
static PetscErrorCode logTdSetup(const CsemSourceSet *sources, const TdParams *td) {
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
  if (td->atsdRtol > 0.0) {
    PetscCall(logKvf(comm, "Step doubling", "trial every %" PetscInt_FMT " steps, rtol %s", td->atsdSteps, formatReal(td->atsdRtol)));
  } else {
    PetscCall(logKvStr(comm, "Step doubling", "off"));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Prints the "Solve" section of the run report. */
static PetscErrorCode logTdSolve(MPI_Comm comm, const TdStats *stats, const TdCharge *charge, PetscReal dt, PetscReal t) {
  PetscFunctionBeginUser;
  PetscCall(logSection(comm, "Solve"));
  PetscCall(logKvStr(comm, "Solver", stats->solver));
  PetscCall(logKvInt(comm, "Time steps", stats->numSteps));
  PetscCall(logKvInt(comm, "Operator setups", stats->numSetups));
  PetscCall(logKvf(comm, "Doublings (acc / rej)", "%" PetscInt_FMT " / %" PetscInt_FMT, stats->numDoubled, stats->numRejected));
  PetscCall(logKvReal(comm, "Last time step (s)", dt));
  PetscCall(logKvReal(comm, "Last time (s)", t));
  PetscCall(logKvInt(comm, "KSP iterations", stats->its));
  PetscCall(logKvInt(comm, "Charge KSP iterations", charge->its));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Prints the header of the time-step table and starts the first row. */
static PetscErrorCode beginTdMonitor(MPI_Comm comm, const TdStats *stats, const TdCharge *charge, TdMonitor *mon) {
  PetscFunctionBeginUser;
  PetscCall(logSection(comm, "Time steps"));
  PetscCall(PetscPrintf(comm, "   %6s %11s %11s %10s %8s %8s %10s %8s\n", "Steps", "t (s)", "dt (s)", "Doubling", "Result", "KSP its",
                        "Charge its", "Wall (s)"));
  mon->its       = stats->its;
  mon->chargeIts = charge->its;
  PetscCall(PetscTime(&mon->tWall));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Prints one row of the time-step table; ratio < 0 means no doubling trial. */
static PetscErrorCode logTdStep(MPI_Comm comm, const TdStats *stats, const TdCharge *charge, PetscReal t, PetscReal dt, PetscReal ratio,
                                const char *result, TdMonitor *mon) {
  PetscFunctionBeginUser;
  PetscLogDouble now;
  char           trial[16];

  if (ratio < 0.0)
    PetscCall(PetscStrncpy(trial, "-", sizeof(trial)));
  else
    PetscCall(PetscSNPrintf(trial, sizeof(trial), "%.3e", (double)ratio));
  PetscCall(PetscTime(&now));
  PetscCall(PetscPrintf(comm, "   %6" PetscInt_FMT " %11.4e %11.4e %10s %8s %8" PetscInt_FMT " %10" PetscInt_FMT " %8.1f\n",
                        stats->numSteps, (double)t, (double)dt, trial, result, stats->its - mon->its, charge->its - mon->chargeIts,
                        (double)(now - mon->tWall)));
  mon->its       = stats->its;
  mon->chargeIts = charge->its;
  mon->tWall     = now;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* s(t) by linear interpolation; constant outside the sampled window. */
static PetscReal evaluateWaveform(const TdParams *td, PetscReal t) {
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
 * @brief Assembles K and G (constFactor 0) and the sigma-weighted mass matrix Ms
 *        on the pattern of K.
 */
static PetscErrorCode assembleTdOperators(const DM dm, const Grid *grid, const Vec conductivity, Mat *K, Mat *Ms, Mat *G) {
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

/* Column-space Vec of G: 0 on the boundary nodal dofs, 1 elsewhere. */
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
 * @brief Builds L = Gi^T Ms Gi (identity on the boundary nodes) and sets up its KSP.
 *
 * Solver: CG + GAMG, rtol 1e-10.
 */
static PetscErrorCode setupTdCharge(const DM dm, const Grid *grid, Mat Ms, Mat G, TdCharge *charge) {
  PetscFunctionBeginUser;

  Mat       MsAij;
  Vec       mask, boundary;
  PC        pc;
  PetscBool isIs;

  PetscCall(buildInteriorNodeMask(dm, grid, G, &mask));
  PetscCall(MatDuplicate(G, MAT_COPY_VALUES, &charge->Gi));
  PetscCall(MatDiagonalScale(charge->Gi, NULL, mask));

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

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)G), &charge->ksp));
  PetscCall(KSPSetOperators(charge->ksp, charge->L, charge->L));
  PetscCall(KSPSetType(charge->ksp, KSPCG));
  PetscCall(KSPGetPC(charge->ksp, &pc));
  PetscCall(PCSetType(pc, PCGAMG));
  PetscCall(KSPSetTolerances(charge->ksp, 1.0e-10, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(KSPSetUp(charge->ksp));

  PetscCall(MatCreateVecs(charge->L, &charge->phi, &charge->rhs));
  PetscCall(MatCreateVecs(Ms, NULL, &charge->h));
  charge->its = 0;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyTdCharge(TdCharge *charge) {
  PetscFunctionBeginUser;
  PetscCall(MatDestroy(&charge->Gi));
  PetscCall(MatDestroy(&charge->L));
  PetscCall(KSPDestroy(&charge->ksp));
  PetscCall(VecDestroy(&charge->h));
  PetscCall(VecDestroy(&charge->rhs));
  PetscCall(VecDestroy(&charge->phi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief E += Gi phi with L phi = -Gi^T (Ms E + s <N, p>), for every source.
 *
 * <N, p> = -B0 / mu.
 */
static PetscErrorCode applyTdCharge(TdCharge *charge, Mat Ms, Mat B0, PetscReal s, Mat E) {
  PetscFunctionBeginUser;

  PetscInt           numRhs, its;
  KSPConvergedReason reason;

  PetscCall(MatGetSize(E, NULL, &numRhs));
  for (PetscInt i = 0; i < numRhs; i++) {
    Vec x, b;
    PetscCall(MatDenseGetColumnVec(E, i, &x));
    PetscCall(MatMult(Ms, x, charge->h));
    if (s != 0.0) {
      PetscCall(MatDenseGetColumnVecRead(B0, i, &b));
      PetscCall(VecAXPY(charge->h, -s / MU, b));
      PetscCall(MatDenseRestoreColumnVecRead(B0, i, &b));
    }
    PetscCall(MatMultTranspose(charge->Gi, charge->h, charge->rhs));
    PetscCall(VecScale(charge->rhs, -1.0));
    PetscCall(KSPSolve(charge->ksp, charge->rhs, charge->phi));
    PetscCall(KSPGetIterationNumber(charge->ksp, &its));
    PetscCall(KSPGetConvergedReason(charge->ksp, &reason));
    PetscCheck(reason > 0, PetscObjectComm((PetscObject)Ms), PETSC_ERR_NOT_CONVERGED, "Charge correction failed: %s",
               KSPConvergedReasons[reason]);
    charge->its += its;
    PetscCall(MatMultAdd(charge->Gi, charge->phi, x, x));
    PetscCall(MatDenseRestoreColumnVec(E, i, &x));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Forms A = K + (a0 mu/dt) Ms and sets up its KSP (factorization or BDDC).
 */
static PetscErrorCode createTdStepper(Mat K, Mat Ms, Mat G, PetscInt order, IS primalVertices, PetscReal a0, PetscReal dt,
                                      TdStepper *stepper, TdStats *stats) {
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyTdStepper(TdStepper *stepper) {
  PetscFunctionBeginUser;
  PetscCall(KSPDestroy(&stepper->ksp));
  PetscCall(MatDestroy(&stepper->A));
  stepper->dt = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief One step from t to t + dt for every source.
 *
 * R = -(mu/dt) Ms (a1 E(t) + a2 E(t-dt)) + (a0 s(t+dt) + a1 s(t) + a2 s(t-dt))/dt B0,
 * with B0 = -mu <N, p>; solves A Enext = R, then applies the charge correction.
 * E1 = E(t-dt) is unused for backward Euler.
 */
static PetscErrorCode advanceTdStep(const TdStepper *stepper, TdCharge *charge, Mat Ms, Mat B0, const TdParams *td, PetscReal t, Mat E0,
                                    Mat E1, Mat Enext, Mat R, TdStats *stats) {
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
  PetscCall(applyTdCharge(charge, Ms, B0, evaluateWaveform(td, t + dt), Enext));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the receiver interpolation; fields are dB/dt in the H slots.
 */
static PetscErrorCode setupTdOutput(const CsemSourceSet sources, const DM dm, const Grid *grid, Vec receivers, TdOutput *out) {
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

/* Appends E and dB/dt = -curl E at the receivers for time t. */
static PetscErrorCode recordTdSample(TdOutput *out, Mat E, PetscReal t) {
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
 * @brief Writes the time series to a SINGLE HDF5 response file.
 *
 *   /                      root attrs: provenance, num_sources, num_times
 *   /times                 sample times (s)
 *   /sources/src{k}/       attrs: x_pos, y_pos, z_pos, current, length,
 *                                 dip_angle, azimuth_angle
 *   /sources/src{k}/fields Ex, Ey, Ez, dBxdt, dBydt, dBzdt, one row per
 *                          receiver and one column per time
 */
static PetscErrorCode writeTdOutput(const PetgemParams *params, const CsemSourceSet sources, const TdOutput *out) {
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
    for (PetscInt n = 0; n < nt; n++) a[n] = out->times[n];
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
        for (PetscInt n = 0; n < nt; n++) a[r * nt + n] = out->history[n * stride + (i * NUM_RECEIVER_FIELDS + comp) * out->numLocal + r];
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

static PetscErrorCode destroyTdOutput(TdOutput *out) {
  PetscFunctionBeginUser;
  PetscCall(destroyReceiverFields(out->fields));
  PetscCall(destroyReceiverInterpolationMatrices(&out->Q));
  PetscCall(PetscFree(out->times));
  PetscCall(PetscFree(out->history));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int runTdCsemForward(int argc, char **argv) {

  if (isVersionRequest(argc, argv)) return printVersion();

  /* Variables declaration */
  KernelRun     run;
  DM            dm;
  Vec           conductivity, materialsId, receivers;
  PetgemParams  params;
  TdParams      td = {0};
  Grid          grid;
  CsemSourceSet sources = {0, 0, NULL};
  Mat           K, Ms, G, B0, E, E1, E2, Enext, Etrial, R, swap;
  IS            primalVertices = NULL;
  TdStepper     current = {0}, trial = {0};
  TdCharge      charge  = {0};
  TdStats       stats   = {0};
  TdMonitor     monitor = {0};
  TdOutput      out;
  PetscReal     t, tSource;
  MPI_Comm      comm;

  PetscFunctionBeginUser;
  PetscCall(beginKernelRun(&argc, &argv, tdHelp, "fm.tdcsem", "Field interpolation", &run));
  comm = PETSC_COMM_WORLD;

  /* Parse user parameters */
  PetscCall(beginPhase(&run, PHASE_PARSE));
  PetscCall(readPetgemParams(&params));
  PetscCall(readTdParams(&td));
  PetscCall(endPhase(&run, PHASE_PARSE));
  if (run.helpRequested) {
    PetscCall(endHelpRun(&run));
    return 0;
  }

  /* Load mesh, conductivity, materials, receivers, sources and waveform */
  PetscCall(beginPhase(&run, PHASE_LOAD));
  PetscCall(loadModelInputs(&params, &dm, &conductivity, &materialsId, &receivers));
  PetscCall(loadTdSettings(&params, &sources, &td));
  PetscCall(endPhase(&run, PHASE_LOAD));

  /* Setup grid */
  PetscCall(beginPhase(&run, PHASE_GRID));
  PetscCall(setupCsemGrid(params.order, &dm, &grid));
  PetscCall(endPhase(&run, PHASE_GRID));
  PetscCall(logGridSummary(params.inputFile, dm, &grid));
  PetscCall(logTdSetup(&sources, &td));

  /* Assembly: K, Ms, G and the dipole term B0 = -mu <N, p> */
  PetscCall(beginPhase(&run, PHASE_ASSEMBLY));
  PetscCall(assembleCsemRhs(sources, dm, &grid, -MU, &B0));
  PetscCall(assembleTdOperators(dm, &grid, conductivity, &K, &Ms, &G));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &E));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &Enext));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &Etrial));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &E1));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &E2));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &R));
  PetscCall(MatZeroEntries(E));
  PetscCall(MatZeroEntries(E1));
  PetscCall(MatZeroEntries(E2));
  PetscCall(endPhase(&run, PHASE_ASSEMBLY));
  PetscCall(logAssembly(comm, K, sources.numSources));

  /* Receiver interpolation */
  PetscCall(beginPhase(&run, PHASE_OUTPUT));
  PetscCall(setupTdOutput(sources, dm, &grid, receivers, &out));
  PetscCall(endPhase(&run, PHASE_OUTPUT));

  /* Time stepping with adaptive step doubling */
  PetscCall(beginPhase(&run, PHASE_SOLVE));
  PetscCall(setupPrimalVerticesFromOptions(dm, K, conductivity, &primalVertices));
  PetscCall(setupTdCharge(dm, &grid, Ms, G, &charge));
  PetscCall(createTdStepper(K, Ms, G, grid.fem.order, primalVertices, td.a[0], td.dt0, &current, &stats));
  t       = td.time[0];
  tSource = td.time[td.numSamples - 1];
  PetscCall(recordTdSample(&out, E, t));
  PetscCall(beginTdMonitor(comm, &stats, &charge, &monitor));
  while (t < td.tEnd) {
    const PetscReal dt     = current.dt;
    PetscReal       ratio  = -1.0;
    const char     *result = "-";

    for (PetscInt k = 0; k < td.atsdSteps && t < td.tEnd; k++) {
      PetscCall(advanceTdStep(&current, &charge, Ms, B0, &td, t, E, E1, Enext, R, &stats));
      swap = E2, E2 = E1, E1 = E, E = Enext, Enext = swap;
      t += current.dt;
      stats.numSteps++;
      PetscCall(recordTdSample(&out, E, t));
    }

    /* Doubling trial: one 2dt step against two dt steps */
    if (td.atsdRtol > 0.0 && t >= tSource && t + 2.0 * current.dt <= td.tEnd) {
      PetscReal diff, norm;

      if (!trial.ksp) PetscCall(createTdStepper(K, Ms, G, grid.fem.order, primalVertices, td.a[0], 2.0 * current.dt, &trial, &stats));
      PetscCall(advanceTdStep(&trial, &charge, Ms, B0, &td, t, E, E2, Etrial, R, &stats));
      for (PetscInt k = 0; k < 2; k++) {
        PetscCall(advanceTdStep(&current, &charge, Ms, B0, &td, t, E, E1, Enext, R, &stats));
        swap = E2, E2 = E1, E1 = E, E = Enext, Enext = swap;
        t += current.dt;
        stats.numSteps++;
        PetscCall(recordTdSample(&out, E, t));
      }
      PetscCall(MatAXPY(Etrial, -1.0, E, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(Etrial, NORM_FROBENIUS, &diff));
      PetscCall(MatCopy(E, Enext, SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(Enext, -1.0, E2, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(Enext, NORM_FROBENIUS, &norm));
      ratio = diff / norm;
      if (diff <= td.atsdRtol * norm) {
        PetscCall(destroyTdStepper(&current));
        current = trial;
        trial   = (TdStepper){0};
        swap = E1, E1 = E2, E2 = swap;
        stats.numDoubled++;
        result = "accepted";
      } else {
        stats.numRejected++;
        result = "rejected";
      }
    }
    PetscCall(logTdStep(comm, &stats, &charge, t, dt, ratio, result, &monitor));
  }
  PetscCall(getSolverName(current.ksp, stats.solver, sizeof(stats.solver)));
  PetscCall(endPhase(&run, PHASE_SOLVE));
  PetscCall(logTdSolve(comm, &stats, &charge, current.dt, t));

  /* Time series at the receivers */
  PetscCall(beginPhase(&run, PHASE_OUTPUT));
  PetscCall(writeTdOutput(&params, sources, &out));
  PetscCall(endPhase(&run, PHASE_OUTPUT));
  PetscCall(logReceiverOutput(comm, run.outputStage, &params, receivers));
  PetscCall(printForwardSummary(&run));

  /* Free memory */
  PetscCall(destroyTdOutput(&out));
  PetscCall(destroyTdStepper(&current));
  PetscCall(destroyTdStepper(&trial));
  PetscCall(destroyTdCharge(&charge));
  PetscCall(ISDestroy(&primalVertices));
  PetscCall(MatDestroy(&K));
  PetscCall(MatDestroy(&Ms));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&B0));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&Enext));
  PetscCall(MatDestroy(&Etrial));
  PetscCall(MatDestroy(&E1));
  PetscCall(MatDestroy(&E2));
  PetscCall(MatDestroy(&R));
  PetscCall(destroyGrid(&grid));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&conductivity));
  PetscCall(VecDestroy(&materialsId));
  PetscCall(VecDestroy(&receivers));
  PetscCall(PetscFree(sources.sourceArray));
  PetscCall(destroyTdParams(&td));

  PetscCall(PetscFinalize());
  return 0;
}
