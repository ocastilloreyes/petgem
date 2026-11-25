/*
 * Filename: tdcsem.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-11-25
 *
 * Description:
 * Time-domain CSEM of fm.tdcsem: time-stepping options and source waveform,
 * operators and solvers of a time step, discrete charge and DC field,
 * receiver time series and run report.
 */

/*
 * Notes:
 * BDF2 (second-order backward differentiation formula, default) or backward
 * Euler discretization of curl curl E + mu sigma dE/dt = -mu dJs/dt with
 * adaptive time step doubling (Cai et al., 2017, Eqs. 5-14). With
 * D = Ms E + s <N, p> and coefficients a = (3/2, -2, 1/2) for BDF2,
 * (1, -1, 0) for backward Euler:
 *   K E(t) + mu/dt (a0 D(t) + a1 D(t-dt) + a2 D(t-2dt)) = 0
 * A doubling to 2dt is accepted when
 *   ||E_2dt - E_dt||_K <= rtol ||E(t+2dt) - E(t)||_K,
 * with ||X||_K^2 = sum_i x_i^T K x_i over the sources; the BDF2 trial step
 * uses E(t) and E(t-2dt).
 * Initial field: zero for s(t0) = 0; otherwise the DC field E = Gi phi with
 * (Gi^T Ms Gi) phi = -s(t0) Gi^T <N, p>, phi = 0 on the boundary nodes.
 * Charge residual: ||Gi^T (Ms E + s(t) <N, p>)|| / ||Gi^T <N, p>||. PEC boundary.
 */

#ifndef TDCSEM_H
#define TDCSEM_H

#include <petsc.h>

#include "csem.h"
#include "grid.h"
#include "io.h"
#include "receiver_interp.h"

/** @brief Time integration scheme. */
typedef enum { TD_SCHEME_BDF2, TD_SCHEME_BE } TdScheme;

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
  Mat       A;   /**< Time-step operator. */
  KSP       ksp; /**< Solver bound to A. */
} TdStepper;

/**
 * @brief Discrete charge: Gi, the nodal operator L = Gi^T Ms Gi of the DC field
 *        and its KSP, and the charge residual.
 */
typedef struct {
  Mat        Gi;      /**< Discrete gradient restricted to the interior nodal columns. */
  Mat        L;       /**< Gi^T Ms Gi, identity on the boundary nodes (step-off only). */
  KSP        ksp;     /**< Solver bound to L, MUMPS Cholesky by default (step-off only). */
  Vec        h;       /**< Work Vec on the H(curl) space. */
  Vec        rhs;     /**< Nodal right-hand side. */
  Vec        phi;     /**< Nodal potential. */
  PetscInt   its;     /**< KSP iterations of the DC field. */
  PetscReal *srcNorm; /**< ||Gi^T <N, p>|| per source. */
  PetscReal  resMax;  /**< Largest charge residual of the run. */
} TdCharge;

/**
 * @brief Time-stepping counters for the run report.
 */
typedef struct {
  PetscInt numSteps;    /**< Accepted time steps. */
  PetscInt numSetups;   /**< Operators set up. */
  PetscInt numCreated;  /**< Operators created (symbolic setup). */
  PetscInt numDoubled;  /**< Accepted doubling trials. */
  PetscInt numRejected; /**< Rejected doubling trials. */
  PetscInt its;         /**< KSP iterations over all solves. */
  char     solver[32];  /**< PC type, or factorization package. */
} TdStats;

/**
 * @brief Counters at the start of the current row of the time-step table.
 */
typedef struct {
  PetscInt       its;   /**< TdStats.its at the row start. */
  PetscLogDouble tWall; /**< Wall-clock time at the row start. */
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

/**
 * @brief Reads the -tdcsem_* options and sets the coefficients of the scheme.
 *
 * The options are checked unless -help is given.
 *
 * @param[out] td  Time-stepping options; the waveform is filled by loadTdSettings().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode readTdParams(TdParams *td);

/**
 * @brief Loads the transmitters and the source waveform from the bundle.
 *
 * Reads /sources (the frequency is not used) and /tdcsem/time,
 * /tdcsem/waveform, checks that the sample times are strictly increasing and
 * that -tdcsem_t_end is after the first sample, and sets the initial time
 * step to the smallest sample spacing when -tdcsem_dt0 is not given.
 *
 * @param[in]     params   PETGEM parameters (input bundle).
 * @param[out]    sources  Transmitter set; sourceArray is allocated here.
 * @param[in,out] td       Time-stepping options; time and waveform are allocated here.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadTdSettings(const PetgemParams *params, CsemSourceSet *sources, TdParams *td);

/**
 * @brief Frees the waveform arrays of the time-stepping options.
 *
 * @param[in,out] td  Time-stepping options.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode destroyTdParams(TdParams *td);

/**
 * @brief Prints the "CSEM source(s)" and "Time stepping" sections of the run report.
 *
 * @param[in] sources  Transmitter set.
 * @param[in] td       Time-stepping options and waveform.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode logTdSetup(const CsemSourceSet *sources, const TdParams *td);

/**
 * @brief Prints the "Solve" section of the run report.
 *
 * @param[in] comm    Communicator of the report.
 * @param[in] stats   Time-stepping counters.
 * @param[in] charge  Discrete charge (DC iterations, largest charge residual).
 * @param[in] dt      Last time step (s).
 * @param[in] t       Last time (s).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode logTdSolve(MPI_Comm comm, const TdStats *stats, const TdCharge *charge, PetscReal dt, PetscReal t);

/**
 * @brief Prints the header of the time-step table and starts the first row.
 *
 * @param[in]  comm   Communicator of the report.
 * @param[in]  stats  Time-stepping counters.
 * @param[out] mon    Counters at the start of the first row.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode beginTdMonitor(MPI_Comm comm, const TdStats *stats, TdMonitor *mon);

/**
 * @brief Prints one row of the time-step table and starts the next one.
 *
 * @param[in]     comm    Communicator of the report.
 * @param[in]     stats   Time-stepping counters.
 * @param[in]     t       Time at the end of the row (s).
 * @param[in]     dt      Time step of the row (s).
 * @param[in]     ratio   ||E_2dt - E_dt||_K / ||E(t+2dt) - E(t)||_K of the doubling trial; < 0 when there is no trial.
 * @param[in]     result  Result of the doubling trial: "accepted", "rejected" or "-".
 * @param[in]     res     Charge residual at t.
 * @param[in,out] mon     Counters at the start of the row.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode logTdStep(MPI_Comm comm, const TdStats *stats, PetscReal t, PetscReal dt, PetscReal ratio, const char *result, PetscReal res,
                         TdMonitor *mon);

/**
 * @brief Normalized source current s(t), by linear interpolation of the
 *        waveform samples; constant outside the sampled window.
 *
 * @param[in] td  Time-stepping options and waveform.
 * @param[in] t   Time (s).
 *
 * @return PetscReal s(t).
 */
PetscReal evaluateWaveform(const TdParams *td, PetscReal t);

/**
 * @brief Assembles the curl-curl matrix K, the discrete gradient G and the
 *        conductivity-weighted mass matrix Ms on the pattern of K.
 *
 * @param[in]  dm            DMPlex carrying the H(curl) section.
 * @param[in]  grid          Grid filled by setupCsemGrid().
 * @param[in]  conductivity  Per-cell conductivity.
 * @param[out] K             Curl-curl matrix.
 * @param[out] Ms            Conductivity-weighted mass matrix.
 * @param[out] G             Discrete gradient.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode assembleTdOperators(const DM dm, const Grid *grid, const Vec conductivity, Mat *K, Mat *Ms, Mat *G);

/**
 * @brief Builds the discrete gradient Gi restricted to the interior nodes and,
 *        for a step-off, the nodal operator L = Gi^T Ms Gi of the DC field.
 *
 * L is the identity on the boundary nodes. Its KSP is MUMPS Cholesky by
 * default, with options prefix tdcsem_dc_.
 *
 * @param[in]  dm      DMPlex carrying the "Boundary" label.
 * @param[in]  grid    Grid with the nodal space H1dm.
 * @param[in]  Ms      Conductivity-weighted mass matrix.
 * @param[in]  G       Discrete gradient.
 * @param[in]  solve   PETSC_TRUE to build L and its KSP (step-off).
 * @param[out] charge  Discrete charge.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode setupTdCharge(const DM dm, const Grid *grid, Mat Ms, Mat G, PetscBool solve, TdCharge *charge);

/**
 * @brief Destroys the objects of the discrete charge.
 *
 * @param[in,out] charge  Discrete charge.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode destroyTdCharge(TdCharge *charge);

/**
 * @brief Charge residual max_i ||Gi^T (Ms E_i + s <N, p>_i)|| / ||Gi^T <N, p>_i||
 *        over the sources i, with <N, p> = -B0 / mu.
 *
 * Updates the largest charge residual of the run.
 *
 * @param[in,out] charge  Discrete charge.
 * @param[in]     Ms      Conductivity-weighted mass matrix.
 * @param[in]     B0      Dipole term -mu <N, p>, one column per source.
 * @param[in]     s       Normalized source current s(t).
 * @param[in]     E       Electric field, one column per source.
 * @param[out]    res     Charge residual.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode computeTdChargeResidual(TdCharge *charge, Mat Ms, Mat B0, PetscReal s, Mat E, PetscReal *res);

/**
 * @brief DC field E = Gi phi with L phi = -s Gi^T <N, p> for every source,
 *        with <N, p> = -B0 / mu.
 *
 * @param[in,out] charge  Discrete charge with L and its KSP.
 * @param[in]     Ms      Conductivity-weighted mass matrix.
 * @param[in]     B0      Dipole term -mu <N, p>, one column per source.
 * @param[in]     s       Normalized source current s(t0).
 * @param[out]    E       DC field, one column per source.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode computeTdDcField(TdCharge *charge, Mat Ms, Mat B0, PetscReal s, Mat E);

/**
 * @brief Forms A = K + (a0 mu/dt) Ms and sets up its KSP (symbolic and numeric
 *        setup).
 *
 * @param[in]     K               Curl-curl matrix.
 * @param[in]     Ms              Conductivity-weighted mass matrix.
 * @param[in]     G               Discrete gradient (PCBDDC only).
 * @param[in]     order           Basis order (PCBDDC only).
 * @param[in]     primalVertices  Primal dofs of PCBDDC, or NULL.
 * @param[in]     a0              Coefficient of D(t) of the scheme.
 * @param[in]     dt              Time step (s).
 * @param[out]    stepper         Operator and KSP.
 * @param[in,out] stats           Time-stepping counters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode createTdStepper(Mat K, Mat Ms, Mat G, PetscInt order, IS primalVertices, PetscReal a0, PetscReal dt, TdStepper *stepper,
                               TdStats *stats);

/**
 * @brief Refills A = K + (a0 mu/dt) Ms on its pattern and sets up its KSP again
 *        (numeric factorization only).
 *
 * @param[in]     K        Curl-curl matrix.
 * @param[in]     Ms       Conductivity-weighted mass matrix.
 * @param[in]     a0       Coefficient of D(t) of the scheme.
 * @param[in]     dt       Time step (s).
 * @param[in,out] stepper  Operator and KSP.
 * @param[in,out] stats    Time-stepping counters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode refillTdStepper(Mat K, Mat Ms, PetscReal a0, PetscReal dt, TdStepper *stepper, TdStats *stats);

/**
 * @brief Destroys the operator and the KSP of a time step.
 *
 * @param[in,out] stepper  Operator and KSP.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode destroyTdStepper(TdStepper *stepper);

/**
 * @brief ||X||_K = (sum_i x_i^T K x_i)^(1/2) over the columns x_i of X.
 *
 * @param[in]  K    Curl-curl matrix.
 * @param[in]  X    Dense matrix, one column per source.
 * @param[out] nrm  ||X||_K.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode normTdCurl(Mat K, Mat X, PetscReal *nrm);

/**
 * @brief One time step from t to t + dt for every source.
 *
 * Solves A Enext = R with
 *   R = -(mu/dt) Ms (a1 E(t) + a2 E(t-dt)) + (a0 s(t+dt) + a1 s(t) + a2 s(t-dt))/dt B0,
 * and B0 = -mu <N, p>. E1 is not used by backward Euler.
 *
 * @param[in]     stepper  Operator and KSP of the time step dt.
 * @param[in]     Ms       Conductivity-weighted mass matrix.
 * @param[in]     B0       Dipole term -mu <N, p>, one column per source.
 * @param[in]     td       Time-stepping options and waveform.
 * @param[in]     t        Current time (s).
 * @param[in]     E0       E(t), one column per source.
 * @param[in]     E1       E(t-dt), one column per source.
 * @param[out]    Enext    E(t+dt), one column per source.
 * @param[out]    R        Work matrix for the right-hand sides.
 * @param[in,out] stats    Time-stepping counters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode advanceTdStep(const TdStepper *stepper, Mat Ms, Mat B0, const TdParams *td, PetscReal t, Mat E0, Mat E1, Mat Enext, Mat R,
                             TdStats *stats);

/**
 * @brief Builds the receiver interpolation and the empty time series.
 *
 * The magnetic slots of the receiver fields hold dB/dt.
 *
 * @param[in]  sources    Transmitter set.
 * @param[in]  dm         DMPlex carrying the H(curl) section.
 * @param[in]  grid       Grid filled by setupCsemGrid().
 * @param[in]  receivers  Receiver coordinates.
 * @param[out] out        Receiver interpolation and time series.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode setupTdOutput(const CsemSourceSet sources, const DM dm, const Grid *grid, Vec receivers, TdOutput *out);

/**
 * @brief Appends E and dB/dt = -curl E at the receivers to the time series.
 *
 * @param[in,out] out  Receiver interpolation and time series.
 * @param[in]     E    Electric field, one column per source.
 * @param[in]     t    Time (s).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode recordTdSample(TdOutput *out, Mat E, PetscReal t);

/**
 * @brief Writes the time series to a single HDF5 response file.
 *
 *   /                      root attrs: provenance, num_sources, num_times
 *   /times                 sample times (s)
 *   /sources/src{k}/       attrs: x_pos, y_pos, z_pos, current, length,
 *                                 dip_angle, azimuth_angle
 *   /sources/src{k}/fields Ex, Ey, Ez, dBxdt, dBydt, dBzdt, one row per
 *                          receiver and one column per time
 *
 * @param[in] params   PETGEM parameters (output directory and stem).
 * @param[in] sources  Transmitter set.
 * @param[in] out      Receiver interpolation and time series.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode writeTdOutput(const PetgemParams *params, const CsemSourceSet sources, const TdOutput *out);

/**
 * @brief Destroys the receiver interpolation and frees the time series.
 *
 * @param[in,out] out  Receiver interpolation and time series.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode destroyTdOutput(TdOutput *out);

#endif /* TDCSEM_H */
