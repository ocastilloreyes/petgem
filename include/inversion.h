/*
 * Filename: inversion.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Data structures and function prototypes for the CSEM inverse
 * kernel (im.csem). Inversion-specific solver functions are also
 * declared here so that solver.c is kept intact for the forward
 * kernel.
 */

#ifndef INVERSION_H
#define INVERSION_H

#include "constants.h"
#include "grid.h"
#include "fem.h" /* Quadrature3D used by InversionContext workspace */
#include "io.h"  /* PetgemParams: shared base parameters embedded below */
#include "receiver_interp.h"
#include "csem.h"

/** @brief simulation_type tag of im.csem products. */
#define PETGEM_SIM_IM "im.csem"

/** @brief Max number of step sizes of the gradient check. */
#define IM_MAX_GRADIENT_CHECK_STEPS 8
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>

/**
 * @brief Inversion source: a transmitter and its frequency.
 */
typedef struct {
  PetscReal  freq;   /**< Frequency (Hz). */
  CsemSource dipole; /**< Transmitter. */
} ImCsemSource;

/**
 * @brief Data component fitted by the inversion.
 *
 * Ex..Hz are field components at the receivers; Zij = Ei / Hj is the ratio
 * of two of them for the same source. Names (imDataComponentNames) are the
 * dataset suffixes of /observed/\<name\> and predicted_\<name\>.
 */
typedef enum {
  IM_DATA_EX,
  IM_DATA_EY,
  IM_DATA_EZ,
  IM_DATA_HX,
  IM_DATA_HY,
  IM_DATA_HZ,
  IM_DATA_ZXX,
  IM_DATA_ZXY,
  IM_DATA_ZYX,
  IM_DATA_ZYY,
  IM_NUM_DATA_COMPONENTS
} ImDataComponent;

/** @brief Dataset names of the data components, indexed by ImDataComponent. */
extern const char *const imDataComponentNames[IM_NUM_DATA_COMPONENTS];

/**
 * @brief Receiver fields of a data component.
 *
 * @param[in]  comp         Data component.
 * @param[out] numerator    Field of a linear component, or Ei of Zij.
 * @param[out] denominator  Hj of Zij, or NUM_RECEIVER_FIELDS for a linear component.
 */
void imDataComponentFields(ImDataComponent comp, ReceiverField *numerator, ReceiverField *denominator);

/**
 * @brief Observed-data source mode for the inverse kernel.
 *
 * Selects which schema loadObservedDataset() reads the [numFreqs x
 * numReceivers] data of each component from:
 *
 *  - OBS_EXTERNAL : the bundle's /observed/\<component\> compound-complex
 *                   datasets (the post-noise external observed data; default).
 *  - OBS_FM_NATIVE: an fm.csem responses HDF5 file's native per-source
 *                   /sources/src{k}/fields/{Ex..Hz} Vecs (source k -> frequency
 *                   row k), from which ratio components are formed. Lets a
 *                   forward run feed the inverse kernel directly, without the
 *                   Python reshape step (noise, if wanted, must already be
 *                   baked into that file).
 */
typedef enum { OBS_EXTERNAL = 0, OBS_FM_NATIVE = 1 } ObservedDataMode;

/**
 * @brief Inversion parameters (read from the PETSc options database).
 */
typedef struct {
  /** Common parameters (input bundle, output paths, basis order), parsed by readPetgemParams(). */
  PetgemParams common;

  PetscInt  maxIter;            /**< Max L-BFGS iterations. */
  PetscInt  lbfgsMemory;        /**< L-BFGS M parameter. */
  PetscReal lambda;             /**< Tikhonov factor. */
  PetscReal errorLevel;         /**< Relative data error. */
  PetscReal gtol;               /**< Gradient convergence tolerance. */
  PetscReal rmsTol;             /**< RMS early-stop (<=0 off). */
  PetscReal diagGradientWeight; /**< Self-weight in the smoother. */

  /** RMS-plateau early stop, consumed by the L-BFGS loop (src/lbfgs.c).
   *  Parsed here with every other im.csem option, rather than read from the
   *  options database at the point of use, so both appear under -help and can
   *  be set from the params file. */
  PetscReal rmsRelTol;      /**< Relative RMS-improvement threshold (0 = off). */
  PetscInt  rmsStallWindow; /**< Consecutive stalled iterations before stopping. */

  /** Fixed material IDs: cells whose materialId matches one of these values
   *  are excluded from gradient smoothing (treated as self-referencing).
   *  Defaults come from the bundle's /im_meta/fixed_materials dataset
   *  (written by the preprocess from sigmas.txt's `fixed` column);
   *  -im_fixed_materials on the CLI is an override. */
  PetscInt numFixedMaterials;                      /**< Number of fixed IDs. */
  PetscInt fixedMaterials[IM_MAX_FIXED_MATERIALS]; /**< Fixed material ID list. */

  /** @{ Provenance flags: PETSC_TRUE iff the field was set on the CLI, so the
   *  bundle reader keeps it. */
  PetscBool   errorLevelFromCli;     /**< Error level came from CLI. */
  PetscBool   fixedMaterialsFromCli; /**< Fixed materials came from CLI. */
  const char *errorLevelOrigin;      /**< "CLI", "bundle" or "default". */
  const char *fixedMaterialsOrigin;  /**< "CLI", "bundle" or "default". */
  /** @} */

  /** Source-frequency entries loaded from the unified bundle's /sources group
   *  (freq, position, current, length, dipAngle, azimuthAngle). Populated by
   *  setupInversionSources. Each entry's frequency lives in imSources[i].freq;
   *  no separate frequency array is kept. */
  PetscInt     numFreqs;                      /**< Number of entries. */
  ImCsemSource imSources[IM_MAX_FREQUENCIES]; /**< One source record per entry. */

  /** VTU snapshot: write the conductivity model every N accepted L-BFGS steps.
   *  0 (default) disables snapshots. Output dir is taken from -output_dir. */
  PetscInt snapshotInterval; /**< Snapshot interval (0 = disabled). */

  /** Observed-data abstraction: which schema/file the misfit data is read
   *  from (see ObservedDataMode). Set by readImParams from
   *  -im_observed_mode. */
  ObservedDataMode observedMode; /**< External vs fm-native source. */

  /** Path to the observed-data file. For OBS_EXTERNAL an empty string means
   *  "use the unified bundle (common.inputFile)"; for OBS_FM_NATIVE it is the
   *  fm.csem responses HDF5 file. Set by -im_observed_file. */
  char observedFile[PETSC_MAX_PATH_LEN]; /**< Observed-data file (empty = bundle). */

  /** Fitted data components, from -im_data_components (default Ex). */
  PetscInt        numComponents;                      /**< Number of components. */
  ImDataComponent components[IM_NUM_DATA_COMPONENTS]; /**< Component list, no repeats. */

  /** -im_gradient_check: compare the adjoint gradient with central differences and stop. */
  PetscBool gradientCheck;                           /**< Run the gradient check instead of the inversion. */
  PetscInt  numCheckSteps;                           /**< Number of step sizes. */
  PetscReal checkSteps[IM_MAX_GRADIENT_CHECK_STEPS]; /**< Step sizes in log(rho), -im_gradient_check_eps. */
} ImParams;

/**
 * @brief Cell-neighbor smoothing graph in CSR format.
 *
 * Row i is owned cell i; neighbors share a vertex with it and are indexed in
 * the smoothing work vector: the overlap=1 local vector on more than one rank,
 * the owned-cell vector otherwise. Fixed cells (materialId in
 * fixedMaterials) reference only themselves and are never updated.
 */
typedef struct {
  PetscInt   numLocalCells;   /**< Number of locally owned cells. */
  PetscBool *isFixed;         /**< PETSC_TRUE for fixed-material cells. */
  PetscInt  *neighborStart;   /**< CSR row pointer, size numLocalCells+1. */
  PetscInt  *neighborList;    /**< Neighbor indices in the work vector. */
  PetscReal *neighborWeights; /**< Normalized 1/dist weights. */
  DM         dmOverlap;       /**< overlap=1 1-DOF/cell DM; NULL when not distributed. */
  Vec        overlapLocal;    /**< Local work vector on dmOverlap. */
  Vec        overlapGlobal;   /**< Global vector on dmOverlap (ghost exchange). */
} NeighborGraph;

/**
 * @brief Per-frequency data, built once and reused by every evaluation.
 */
typedef struct {
  PetscScalar iwmu;                         /**< i omega mu. */
  Vec         B;                            /**< Source right-hand side (global, on dm). */
  Vec         dObs[IM_NUM_DATA_COMPONENTS]; /**< Observed data per fitted component (sequential). */
  Vec         W[IM_NUM_DATA_COMPONENTS];    /**< Data weights 1 / (|dObs| errorLevel), 0 for missing data (sequential). */
} InversionFrequency;

/**
 * @brief State of an inversion run, shared by the objective/gradient and the L-BFGS monitor.
 *
 * The workspace block is built by setupInversionWorkspace before the
 * optimization and does not depend on the iterate.
 */
typedef struct {
  const ImParams                      *imParams;           /**< Inversion parameters. */
  DM                                   dm;                 /**< H(curl) DM. */
  DM                                   dmConductivity;     /**< Conductivity DM (3 DOF/cell). */
  DM                                   dmInversion;        /**< Inversion DM (1 DOF/cell). */
  Grid                                 grid;               /**< Finite-element grid descriptor. */
  Vec                                  conductivity;       /**< Current sigma model (local). */
  Vec                                  X0;                 /**< Initial log(rho) (global). */
  Vec                                  DfDm;               /**< Gradient accumulator (local). */
  const ReceiverInterpolationMatrices *Q;                  /**< Receiver-interpolation operators. */
  const NeighborGraph                 *graph;              /**< Neighbor smoothing graph. */
  Vec                                  notFixedMaskGlobal; /**< 0 at fixed cells, 1 elsewhere (global). */
  Vec                                  notFixedMaskLocal;  /**< Same mask, local. */

  PetscReal         *allRms;       /**< RMS per evaluation. */
  PetscReal         *gradNorms;    /**< ||g|| per accepted iteration (0 = initial model). */
  PetscReal          lastRms;      /**< RMS of the last evaluation. */
  PetscReal          lastRegTerm;  /**< Tikhonov term of the last evaluation. */
  PetscInt           numEvals;     /**< Objective-gradient evaluations. */
  PetscInt           acceptedIter; /**< Accepted L-BFGS steps. */
  PetscLogDouble     tAssembly;    /**< Ms refill + A_f = K - i omega mu Ms (s). */
  PetscLogDouble     tSolver;      /**< Forward + adjoint solves (s). */
  PetscLogDouble     tOutput;      /**< VTU snapshots (s). */
  PetscLogDouble     tRow;         /**< Time of the last L-BFGS table row. */
  PetscInt           numSolves;    /**< Forward + adjoint solves. */
  KSPConvergedReason failReason;   /**< First negative converged reason, 0 if none. */
  PetscInt           kspItsMinRow; /**< Fewest KSP iterations since the last row. */
  PetscInt           kspItsMaxRow; /**< Most KSP iterations since the last row. */
  PetscInt           numSnapshots; /**< VTU snapshots written. */

  /* Workspace */
  Quadrature3D        quad3d;                      /**< 3D quadrature rule. */
  PetscReal         **Me, **Ke;                    /**< Elemental matrices (numDofInCell^2). */
  Vec                 x, nB, nx;                   /**< Forward solution, adjoint RHS, adjoint solution (global, on dm). */
  Vec                 recv[NUM_RECEIVER_FIELDS];   /**< Fields at the receivers, H divided by i omega mu; NULL if unused (parallel). */
  Vec                 adjSrc[NUM_RECEIVER_FIELDS]; /**< Adjoint source per field, J^T W^2 conj(dObs - d) (parallel, like recv). */
  Vec                 dPre;                        /**< Predicted data of one component (parallel, like recv). */
  PetscInt            numData;                     /**< Real data count: 2 x observed values that are not missing. */
  Mat                 K;                           /**< Curl-curl stiffness (sigma-independent). */
  Mat                 Ms;                          /**< Sigma-weighted mass matrix, refilled every evaluation. */
  Mat                 G;                           /**< Discrete gradient for PCBDDC. */
  Mat                 A;                           /**< A_f = K - i omega mu Ms, refilled for every frequency. */
  KSP                 ksp;                         /**< Solver bound to A (forward and adjoint solves of every frequency). */
  PetscInt            numFreqs;                    /**< Number of frequencies. */
  InversionFrequency *freq;                        /**< Per-frequency data and solvers. */
} InversionContext;

/* ------------------------------------------------------------------ */
/* Function prototypes                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Reads inversion parameters from the PETSc options database.
 *
 * @param[out] imParams  Struct receiving the parsed inversion parameters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode readImParams(ImParams *imParams);

/**
 * @brief Applies case-property defaults from the bundle to imParams.
 *
 * Reads the error_level attribute on /observed and the fixed_materials array
 * under /im_meta and applies them UNLESS the corresponding CLI override was
 * present (see the *FromCLI provenance flags). Safe to call even when the
 * bundle has no such entries - imParams keeps the readImParams defaults.
 *
 * @param[in]     bundleFile  Path to the unified PETGEM HDF5 bundle.
 * @param[in,out] imParams   Inversion parameters updated in place.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode loadInversionMetaFromBundle(const char *bundleFile, ImParams *imParams);

/**
 * @brief Loads multi-frequency inversion sources from the unified bundle.
 *
 * Reads the bundle's /sources group
 * and populates imParams->numFreqs and imSources[]. `bundleFile` is the same
 * HDF5 path consumed by loadModelInputs.
 *
 * @param[in]     bundleFile  Path to the unified PETGEM HDF5 bundle.
 * @param[in,out] imParams   Inversion parameters whose sources are filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode setupInversionSources(const char *bundleFile, ImParams *imParams);

/**
 * @brief Prints the "Inversion sources" and "Inversion parameters" sections
 *        of the im.csem run report.
 *
 * @param[in] imParams  Inversion parameters, after setupInversionSources()
 *                       and loadInversionMetaFromBundle().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode logInversionSummary(const ImParams *imParams);

/**
 * @brief Observed-data abstraction entry point used by the inverse kernel.
 *
 * Reads /observed/\<component\> of the bundle (OBS_EXTERNAL) or the per-source
 * /sources/src{k}/fields of an fm.csem responses file (OBS_FM_NATIVE) for every
 * fitted component, NaN marking missing data,
 * from imParams->observedFile or, when empty, imParams->common.inputFile. Every
 * rank receives the full array.
 *
 * @param[in]  imParams       Inversion parameters (mode, file, numFreqs, components, bundle).
 * @param[in]  numReceivers  Number of receivers (columns).
 * @param[out] dObs          Observed data, component-major numComponents x numFreqs x numReceivers (caller frees).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode loadObservedDataset(const ImParams *imParams, PetscInt numReceivers, PetscScalar **dObs);

/**
 * @brief Builds the CSR neighbor smoothing graph from DMPlex topology.
 *
 * Cells whose materialId matches any entry in imParams->fixedMaterials are
 * treated as fixed (self-referencing only, excluded from smoothing). On more
 * than one rank the graph spans one layer of ghost cells (vertex adjacency).
 *
 * @param[in]  dm           DMPlex mesh.
 * @param[in]  grid         Finite-element grid descriptor.
 * @param[in]  imParams      Inversion parameters (fixed-material list).
 * @param[in]  materialsId  Per-cell material-id Vec.
 * @param[out] graph        Neighbor graph to populate.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode buildNeighborSmoothingGraph(const DM dm, const Grid *grid, const ImParams *imParams, Vec materialsId, NeighborGraph *graph);

/**
 * @brief Applies two Jacobi smoothing sweeps to a 1-DOF/cell local vector.
 *
 * Each sweep reads a frozen copy, with a ghost exchange before each sweep on
 * more than one rank, so the result does not depend on the partition.
 *
 * @param[in]     graph      Neighbor smoothing graph.
 * @param[in]     diagWeight Self-weight applied to the diagonal during sweeps.
 * @param[in,out] v          Vector smoothed in place.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode applyGraphSmoothing(const NeighborGraph *graph, PetscReal diagWeight, Vec v);

/**
 * @brief Recovers conductivity from a log-conductivity perturbation.
 *
 * Computes sigma = 1 / exp(X_smooth + X0) (isotropic, written to components
 * 0-2). X and X0 are global Vecs on dmInversion (1 DOF/cell); sigmaModel is a
 * local Vec on dmConductivity. Graph smoothing is applied to a local
 * copy of X (zero self-weight) before exp; X is not modified.
 *
 * @param[in]  dmInversion     DM for the inversion field (1 DOF/cell).
 * @param[in]  dmConductivity  DM for the conductivity field.
 * @param[in]  X               Log-perturbation iterate (global).
 * @param[in]  X0              Initial log(rho) (global).
 * @param[out] sigmaModel      Recovered conductivity (local).
 * @param[in]  grid            Finite-element grid descriptor.
 * @param[in]  graph           Neighbor smoothing graph.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode applyLogToSigma(DM dmInversion, DM dmConductivity, const Vec X, const Vec X0, Vec sigmaModel, const Grid *grid,
                               const NeighborGraph *graph);

/**
 * @brief Writes a VTU snapshot of the current conductivity model (ρ = 1/σ).
 *
 * Called after accepted L-BFGS steps when snapshotInterval > 0; the output
 * file is {output_dir}/{output_filename}_iter{N:05d}.pvtu, with one
 * {output_filename}_iter{N:05d}_p{rank:04d}.vtu piece per rank - the same
 * output stem the kernel's .h5 product uses.
 *
 * @param[in] ctx           Inversion context.
 * @param[in] acceptedIter  Index of the accepted iteration (used in filename).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode writeInversionSnapshotVtu(const InversionContext *ctx, PetscInt acceptedIter);

/**
 * @brief Wall times and counters of an inversion run, for the run report.
 */
typedef struct {
  PetscLogDouble     tSetup;       /**< Operators, RHS, receivers, observed data, KSPs. */
  PetscLogDouble     tAssembly;    /**< Ms refill + A_f = K - iωμ·Ms. */
  PetscLogDouble     tSolver;      /**< Forward + adjoint solves. */
  PetscLogDouble     tGradient;    /**< Rest of the L-BFGS loop (gradient, smoothing, updates). */
  PetscLogDouble     tOutput;      /**< VTU snapshots + HDF5 results. */
  PetscInt           numSolves;    /**< Forward + adjoint solves. */
  KSPConvergedReason failReason;   /**< First negative converged reason, 0 if none. */
  char               solver[32];   /**< Solver name (getSolverName). */
  PetscInt           numSnapshots; /**< VTU snapshots written. */
} InversionStats;

/**
 * @brief Top-level inversion driver.
 *
 * `receivers` is the serial Vec (PETSC_COMM_SELF, length 3·N_recv) loaded by
 * loadModelInputs from /receivers in the unified PETGEM input HDF5. It is
 * consumed once by buildReceiverInterpolationMatrices; the caller retains
 * ownership for cleanup.
 *
 * @param[in]  imParams       Inversion parameters.
 * @param[in]  dm            H(curl) DM.
 * @param[in]  grid          Finite-element grid descriptor.
 * @param[in]  conductivity  Initial per-cell conductivity Vec.
 * @param[in]  materialsId   Per-cell material-id Vec.
 * @param[in]  receivers     Serial Vec of 3·N_recv receiver coordinates.
 * @param[out] stats         Wall times and counters for the run report.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode runCsemInversion(const ImParams *imParams, const DM dm, const Grid *grid, Vec conductivity, Vec materialsId, Vec receivers,
                                InversionStats *stats);

/**
 * @brief Frees the NeighborGraph memory.
 *
 * @param[in,out] graph  Neighbor graph to destroy.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode destroyNeighborGraph(NeighborGraph *graph);

#endif /* INVERSION_H */
