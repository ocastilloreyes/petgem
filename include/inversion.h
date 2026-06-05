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
#include "hvfem.h"        /* Quadrature3D used by InversionContext workspace */
#include "inputs.h"       /* fmParams: shared base parameters embedded below */
#include "receiver_interp.h"
#include "transmitter.h"
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
/**
 * @brief Inversion source: a transmitter carrying its own frequency.
 *
 * Extends the forward CsemSource layout with a per-source frequency so a
 * multi-frequency inversion can store one record per (source, frequency).
 */
typedef struct {
  PetscReal freq;           /**< Frequency (Hz). */
  PetscReal position[3];    /**< Transmitter position (x, y, z). */
  PetscReal current;        /**< Electric current. */
  PetscReal length;         /**< Dipole length. */
  PetscReal dipAngle;       /**< Dip angle. */
  PetscReal azimuthAngle;   /**< Azimuth angle. */
} InvCsemSource;

/**
 * @brief Observed-data source mode for the inverse kernel.
 *
 * Selects which schema loadObservedDataset() reads the [numFreqs x
 * numReceivers] Ex misfit data from:
 *
 *  - OBS_EXTERNAL : the bundle's /observed/Ex compound-complex dataset
 *                   (the post-noise external observed data; default).
 *  - OBS_FM_NATIVE: an fm.csem responses HDF5 file's native per-source
 *                   /sources/src{k}/fields/Ex Vecs (source k -> frequency
 *                   row k). Lets a forward run feed the inverse kernel
 *                   directly, without the Python reshape step (noise, if
 *                   wanted, must already be baked into that file).
 */
typedef enum {
  OBS_EXTERNAL  = 0,
  OBS_FM_NATIVE = 1
} ObservedDataMode;

/**
 * @brief Inversion parameters (read from the PETSc options database).
 */
typedef struct {
  /** Shared base parameters, parsed by the SAME readfmParams() the forward
   *  kernel uses (input/output paths, basis order, MPI task count, quiet
   *  flag). Unifies the fm.csem / im.csem interface: the inverse-only
   *  controls below extend this common base. The basis order is `fm.nord`
   *  and the unified-bundle path (consumed by setupInversionSources,
   *  loadInversionMetaFromBundle, loadObservedData) is `fm.inputFile` -
   *  formerly the separate `bundleFile` mirror. */
  fmParams  fm;

  PetscInt  maxIter;                               /**< Max L-BFGS iterations. */
  PetscInt  lbfgsMemory;                           /**< L-BFGS M parameter. */
  PetscReal lambda;                                /**< Tikhonov factor. */
  PetscReal errorLevel;                            /**< Relative data error. */
  PetscReal gtol;                                  /**< Gradient convergence tolerance. */
  PetscReal rmsTol;                                /**< RMS early-stop (<=0 off). */
  PetscReal diagGradientWeight;                    /**< Self-weight in the smoother. */

  /** Fixed material IDs: cells whose material_id matches one of these values
   *  are excluded from gradient smoothing (treated as self-referencing).
   *  Defaults come from the bundle's /inv_meta/fixed_materials dataset
   *  (written by the preprocess from sigmas.txt's `fixed` column);
   *  -inv_fixed_materials on the CLI is an override. */
  PetscInt  numFixedMaterials;                     /**< Number of fixed IDs. */
  PetscInt  fixedMaterials[INV_MAX_FIXED_MATERIALS]; /**< Fixed material ID list. */

  /** @{ Provenance flags: PETSC_TRUE iff the field was set on the CLI (and so
   *  should NOT be overridden by the bundle reader). */
  PetscBool errorLevelFromCLI;                     /**< Error level came from CLI. */
  PetscBool fixedMaterialsFromCLI;                 /**< Fixed materials came from CLI. */
  /** @} */

  /** Source-frequency entries loaded from the unified bundle's /sources group
   *  (freq, position, current, length, dipAngle, azimuthAngle). Populated by
   *  setupInversionSources. Each entry's frequency lives in invSources[i].freq;
   *  no separate frequency array is kept. */
  PetscInt       numFreqs;                         /**< Number of entries. */
  InvCsemSource  invSources[INV_MAX_FREQUENCIES];  /**< One source record per entry. */

  /** VTU snapshot: write the conductivity model every N accepted L-BFGS steps.
   *  0 (default) disables snapshots. Output dir is taken from -output_dir. */
  PetscInt  snapshotInterval;                      /**< Snapshot interval (0 = disabled). */

  /** Observed-data abstraction: which schema/file the misfit data is read
   *  from (see ObservedDataMode). Set by readInversionParams from
   *  -inv_observed_mode. */
  ObservedDataMode observedMode;                   /**< External vs fm-native source. */

  /** Path to the observed-data file. For OBS_EXTERNAL an empty string means
   *  "use the unified bundle (fm.inputFile)"; for OBS_FM_NATIVE it is the
   *  fm.csem responses HDF5 file. Set by -inv_observed_file. */
  char      observedFile[PETSC_MAX_PATH_LEN];      /**< Observed-data file (empty = bundle). */
} imParams;

/**
 * @brief Neighbor smoothing graph in CSR format.
 *
 * Fixed elements (material_id matching any entry in fixedMaterials) are
 * flagged so they are excluded from gradient smoothing.
 */
typedef struct {
  PetscInt   numLocalCells;    /**< Number of locally owned cells. */
  PetscInt  *neighborStart;    /**< CSR row pointer, size numLocalCells+1. */
  PetscInt  *neighborList;     /**< Local cell IDs of neighbors. */
  PetscReal *neighborWeights;  /**< Normalized 1/dist weights. */
  PetscBool *isFixed;          /**< PETSC_TRUE for fixed-material cells. */

  /** Fully-parallel block-Jacobi GS state, set by setupParallelSmoothingGraph.
   *  When hasParallelGraph is true (production path for MPI > 1),
   *  applyGaussSeidelSmoothing runs forward + reverse Gauss-Seidel on each
   *  rank's owned cells using its own local copy plus one layer of ghost
   *  values, exchanging ghosts between sweeps (DMLocalToGlobal /
   *  DMGlobalToLocal pair). Cost is O(local cells) per smoother call, no
   *  rank-0 bottleneck. NULL/false on single-rank runs (the local sweep in
   *  applyGaussSeidelSmoothing handles serial directly). */
  PetscBool  hasParallelGraph; /**< True when the parallel GS state is built. */
  DM         dmInversionOver;  /**< overlap=1 1-DOF/cell DM (owned). */
  PetscInt  *oNeighborStart;   /**< CSR row pointer [numLocalCells+1]. */
  PetscInt  *oNeighborList;    /**< Local overlap-1 neighbor indices. */
  PetscReal *oNeighborWeights; /**< Normalized 1/dist weights. */
  Vec        oLocalScratch;    /**< Local Vec on dmInversionOver. */
  Vec        oGlobalScratch;   /**< Global Vec on dmInversionOver. */
} NeighborGraph;

/**
 * @brief Context passed to the L-BFGS objective/gradient callback.
 *
 * Bundles the DMs, model/gradient Vecs, observed data, precomputed
 * per-frequency inputs, and cached matrices/solvers reused across every
 * objgrad evaluation. The pre-allocated workspace fields are owned by
 * setupInversionWorkspace and freed by destroyInversionWorkspace.
 */
typedef struct {
  const imParams                *iparams;        /**< Inversion parameters. */
  DM                              dm;            /**< H(curl) DM. */
  DM                              dmConductivity; /**< DM for conductivity (3 DOF/cell). */
  DM                              dmInversion;    /**< DM for inversion (1 DOF/cell). */
  Grid                            grid;          /**< Finite-element grid descriptor. */
  Vec                             conductivity; /**< Current sigma model (local). */
  Vec                             X0;          /**< Initial log(rho) (global, 1 DOF/cell). */
  Vec                             DfDm;        /**< Gradient workspace (local, 1 DOF/cell). */
  Mat                             dObs;        /**< Observed Ex, Nfreq × Nrec dense. */
  const ReceiverInterpolationMatrices *Q;      /**< Receiver-interpolation operators. */
  const NeighborGraph            *graph;       /**< Neighbor smoothing graph. */
  Vec                             notFixedMaskGlobal; /**< 0 at fixed cells, 1 elsewhere (global, dmInversion). */
  Vec                             notFixedMaskLocal;  /**< Same as above but local (dmInversion). */
  PetscReal                      *allRMS;      /**< Output: RMS per iteration. */
  PetscReal                       lastRMS;     /**< RMS at most-recent callback call. */
  PetscReal                       lastDataMisfit; /**< Data term at most-recent eval. */
  PetscReal                       lastRegTerm;    /**< Tikhonov term at most-recent eval. */
  PetscInt                        iterCount;   /**< Total objgrad calls. */
  PetscInt                        acceptedIter;/**< Accepted L-BFGS steps. */

  /** @{ Phase timers accumulated across all objgrad evaluations (seconds), so
   *  im.csem can report an Assembly/Solver breakdown consistent with fm.csem
   *  instead of lumping the whole inversion into one bucket. */
  PetscLogDouble                  tAssembly;   /**< Ms refill + A = K - iωμ·Ms. */
  PetscLogDouble                  tSolver;     /**< Factorize + fwd/adjoint solves. */
  /** @} */

  /* ---- Pre-allocated workspace ----
   * Owned by setupInversionWorkspace; freed by destroyInversionWorkspace.
   * All fields below are populated once before the L-BFGS loop and reused
   * across every objgrad evaluation. Their contents do not depend on the
   * iterate X, so precomputing yields byte-identical numerical results to
   * recomputing on every callback. */

  /** @{ Per-cell elemental-matrix work buffers used by
   *  computeGradientContribution (numDofInCell × numDofInCell each, zeroed
   *  per cell inside the loop). */
  PetscReal                      *MeBuf;        /**< Contiguous backing (numDof²). */
  PetscReal                      *KeBuf;        /**< Contiguous backing (numDof²). */
  PetscReal                     **MeRows;       /**< Row-of-pointers view of MeBuf. */
  PetscReal                     **KeRows;       /**< Row-of-pointers view of KeBuf. */
  /** @} */
  /** 3D quadrature for elemental mass-matrix integration. Depends only on
   *  iparams->fm.nord (constant across the run). */
  Quadrature3D                    quad3d;       /**< 3D quadrature rule. */
  PetscBool                       quad3dInited; /**< PETSC_TRUE once quad3d is filled. */

  /** @{ Per-iteration reusable Vecs (global, sized on dm). DMCreateGlobalVector
   *  caches storage on the DM, but moving them out of the callback still saves
   *  the 4× DMCreateGlobalVector + 2× VecCreate calls per iter and keeps the
   *  same buffers warm across iterations. */
  Vec                             bVec;        /**< RHS workspace (forward solve). */
  Vec                             xVec;        /**< Forward solution. */
  Vec                             nBvec;       /**< Adjoint RHS. */
  Vec                             nxVec;       /**< Adjoint solution. */
  Vec                             ExRecvVec;   /**< Ex at receivers, parallel. */
  Vec                             wcdtDvec;    /**< Weighted residual workspace, parallel. */
  /** @} */

  /** @{ Per-frequency precomputed inputs. The RHS Vec, weights Vec, and
   *  observed-Ex row depend only on the source, the observed data, and
   *  iparams->errorLevel - all constant across L-BFGS iterations. */
  Vec                            *Bvec_per_freq;     /**< numFreqs, sized on dm. */
  Vec                            *Wf_per_freq;       /**< numFreqs, sized seq Nrec. */
  Vec                            *dObsRow_per_freq;  /**< numFreqs, sized seq Nrec. */
  PetscInt                        numFreqsAlloc;     /**< Size of the arrays above. */
  /** @} */

  /** @{ Cached LHS matrices. K (stiffness, σ-independent) and G_BDDC
   *  (topological vertex incidence, σ-independent) are built ONCE at setup via
   *  assembleCsemKandM and reused across all L-BFGS iters. Ms keeps the
   *  sparsity from that build but its values are refilled every callback via
   *  assembleCsemMsRefill against the current σ. */
  Mat                             Kmat;        /**< Curl-curl stiffness (σ-independent). */
  Mat                             Msmat;       /**< Mass-σ matrix (refilled per callback). */
  Mat                             Gmat_BDDC;   /**< Topological discrete gradient (BDDC hint). */
  /** @} */

  /** @{ Persistent per-frequency system matrices and solvers. The sparsity
   *  pattern of A_f = K - iωμ·Ms is invariant across both frequency and L-BFGS
   *  iteration (SAME_NONZERO_PATTERN), so each Avec_per_freq[f] is allocated
   *  ONCE and refilled in place every objgrad evaluation, and each
   *  ksp_per_freq[f] is created ONCE bound to it. Reusing the KSP keeps the
   *  (expensive) symbolic factorization / BDDC topological setup across all
   *  iterations; only the cheap numeric refactorization is redone when the
   *  matrix values change. Created by setupInversionWorkspace, freed by
   *  destroyInversionWorkspace. */
  Mat                            *Avec_per_freq;     /**< numFreqs, sized on dm. */
  KSP                            *ksp_per_freq;      /**< numFreqs. */
  /** @} */
} InversionContext;

/* ------------------------------------------------------------------ */
/* Function prototypes                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Reads inversion parameters from the PETSc options database.
 *
 * @param[out] iparams  Struct receiving the parsed inversion parameters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode readInversionParams(imParams *iparams);

/**
 * @brief Applies case-property defaults from the bundle to iparams.
 *
 * Reads the error_level attribute on /observed and the fixed_materials array
 * under /inv_meta and applies them UNLESS the corresponding CLI override was
 * present (see the *FromCLI provenance flags). Safe to call even when the
 * bundle has no such entries - iparams keeps the readInversionParams defaults.
 *
 * @param[in]     bundleFile  Path to the unified PETGEM HDF5 bundle.
 * @param[in,out] iparams     Inversion parameters updated in place.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode loadInversionMetaFromBundle(const char *bundleFile,
                                            imParams  *iparams);

/**
 * @brief Loads multi-frequency inversion sources from the unified bundle.
 *
 * Reads the bundle's /sources group (replacing the legacy text-file format)
 * and populates iparams->numFreqs and invSources[]. `bundleFile` is the same
 * HDF5 path consumed by loadCsemInputs.
 *
 * @param[in]     bundleFile  Path to the unified PETGEM HDF5 bundle.
 * @param[in,out] iparams     Inversion parameters whose sources are filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode setupInversionSources(const char *bundleFile,
                                     imParams  *iparams);

/**
 * @brief Loads observed data from the unified bundle's /observed/Ex dataset.
 *
 * Reads the HDF5 compound complex128 dataset of shape
 * [numFreqs, numReceivers] into a dense Mat on PETSC_COMM_SELF. This is the
 * OBS_EXTERNAL backend of loadObservedDataset().
 *
 * @param[in]  bundleFile    Path to the unified PETGEM HDF5 bundle.
 * @param[in]  numFreqs      Number of frequencies (rows).
 * @param[in]  numReceivers  Number of receivers (columns).
 * @param[out] dObs          Dense Mat (numFreqs × numReceivers) of observed Ex.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode loadObservedData(const char *bundleFile,
                                PetscInt    numFreqs,
                                PetscInt    numReceivers,
                                Mat        *dObs);

/**
 * @brief Loads observed Ex from an fm.csem responses file's native schema.
 *
 * OBS_FM_NATIVE backend of loadObservedDataset(): reads the per-source
 * /sources/src{k}/fields/Ex PETSc Vecs written by the forward kernel
 * (k = 1..numFreqs maps to frequency rows 0..numFreqs-1) into the same
 * [numFreqs × numReceivers] dense Mat layout the external path produces, so
 * the inverse kernel is agnostic to the data origin. Each rank reads the
 * COMM_SELF Vecs independently (no broadcast).
 *
 * @param[in]  responsesFile  Path to the fm.csem responses HDF5 file.
 * @param[in]  numFreqs       Number of source/frequency rows.
 * @param[in]  numReceivers   Number of receivers (Ex length / columns).
 * @param[out] dObs           Dense Mat (numFreqs × numReceivers) of observed Ex.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode loadObservedFmNative(const char *responsesFile,
                                    PetscInt    numFreqs,
                                    PetscInt    numReceivers,
                                    Mat        *dObs);

/**
 * @brief Observed-data abstraction entry point used by the inverse kernel.
 *
 * Dispatches to loadObservedData() (OBS_EXTERNAL) or loadObservedFmNative()
 * (OBS_FM_NATIVE) according to iparams->observedMode, resolving the file
 * path from iparams->observedFile (falling back to the unified bundle
 * iparams->fm.inputFile when empty). numFreqs is taken from
 * iparams->numFreqs.
 *
 * @param[in]  iparams       Inversion parameters (mode, file, numFreqs, bundle).
 * @param[in]  numReceivers  Number of receivers (columns).
 * @param[out] dObs          Dense Mat (numFreqs × numReceivers) of observed Ex.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode loadObservedDataset(const imParams *iparams,
                                   PetscInt        numReceivers,
                                   Mat            *dObs);

/**
 * @brief Builds the CSR neighbor smoothing graph from DMPlex topology.
 *
 * Cells whose material_id matches any entry in iparams->fixedMaterials are
 * treated as fixed (self-referencing only, excluded from smoothing).
 *
 * @param[in]  dm           DMPlex mesh.
 * @param[in]  grid         Finite-element grid descriptor.
 * @param[in]  iparams      Inversion parameters (fixed-material list).
 * @param[in]  materialsID  Per-cell material-id Vec.
 * @param[out] graph        Neighbor graph to populate.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode buildNeighborSmoothingGraph(const DM dm,
                                           const Grid *grid,
                                           const imParams *iparams,
                                           Vec materialsID,
                                           NeighborGraph *graph);

/**
 * @brief Accumulates the per-element adjoint gradient into DfDm (1 DOF/cell).
 *
 * Computes DfDm[ie] += real( (-2·constFactor·Me_e·x_e)^T · nx_e ) using a
 * plain transpose (no conjugate), matching MATLAB iG.'*inx. `quadrature_3d`,
 * `Me`, `Ke` are caller-owned workspace buffers (set up once on
 * InversionContext via setupInversionWorkspace).
 *
 * @param[in]     dm             H(curl) DM.
 * @param[in]     grid           Finite-element grid descriptor.
 * @param[in]     conductivity   Current conductivity Vec.
 * @param[in]     xLocal         Forward solution (local).
 * @param[in]     nxLocal        Adjoint solution (local).
 * @param[in]     constFactor    Frequency factor iωμ.
 * @param[in]     dmInversion    DM for the inversion field (1 DOF/cell).
 * @param[in,out] DfDm           Gradient accumulator (local, 1 DOF/cell).
 * @param[in]     quadrature_3d  3D quadrature workspace.
 * @param[in,out] Me             Scratch elemental mass buffer.
 * @param[in,out] Ke             Scratch elemental stiffness buffer.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode computeGradientContribution(const DM dm,
                                           const Grid *grid,
                                           const Vec conductivity,
                                           const Vec xLocal,
                                           const Vec nxLocal,
                                           PetscScalar constFactor,
                                           DM dmInversion,
                                           Vec DfDm,
                                           const Quadrature3D *quadrature_3d,
                                           PetscReal **Me,
                                           PetscReal **Ke);

/**
 * @brief Applies forward + reverse Gauss-Seidel smoothing to a vector.
 *
 * On serial / single-rank runs this gathers to rank 0, sweeps, and scatters
 * back; the parallel block-Jacobi path is used when the graph carries the
 * overlap-1 state (see NeighborGraph::hasParallelGraph).
 *
 * @param[in]     graph      Neighbor smoothing graph.
 * @param[in]     diagWeight Self-weight applied to the diagonal during sweeps.
 * @param[in,out] v          Vector smoothed in place.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode applyGaussSeidelSmoothing(const NeighborGraph *graph,
                                         PetscReal diagWeight,
                                         Vec v);

/**
 * @brief Recovers conductivity from a log-conductivity perturbation.
 *
 * Computes sigma = 1 / exp(X_smooth + X0) (isotropic, written to components
 * 0-2). X and X0 are global Vecs on dmInversion (1 DOF/cell); sigmaModel is a
 * local Vec on dmConductivity. Gauss-Seidel smoothing is applied to a local
 * copy of X before exp, matching MATLAB's tempX smoothing after each L-BFGS step.
 *
 * @param[in]  dmInversion     DM for the inversion field (1 DOF/cell).
 * @param[in]  dmConductivity  DM for the conductivity field.
 * @param[in]  X               Log-perturbation iterate (global).
 * @param[in]  X0              Initial log(rho) (global).
 * @param[out] sigmaModel      Recovered conductivity (local).
 * @param[in]  grid            Finite-element grid descriptor.
 * @param[in]  graph           Neighbor smoothing graph.
 * @param[in]  diagWeight      Smoother self-weight.
 * @param[out] xPostSmoothOut  Optional (may be NULL): local Vec on dmInversion
 *                             receiving the smoothed X (pre-X0 add) for VTU
 *                             diagnostic snapshots.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode applyLogToSigma(DM dmInversion, DM dmConductivity,
                                const Vec X, const Vec X0,
                                Vec sigmaModel, const Grid *grid,
                                const NeighborGraph *graph,
                                PetscReal diagWeight,
                                Vec xPostSmoothOut);

/**
 * @brief Creates a DM with 1 DOF per cell for the inversion variables.
 *
 * Clones the DMPlex topology from dmConductivity but sets a new section with
 * a single scalar DOF per cell (for X, X0, gradient).
 *
 * @param[in]  dmConductivity  DM whose topology is cloned.
 * @param[in]  grid            Finite-element grid descriptor.
 * @param[out] dmInv           New 1-DOF/cell inversion DM.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode createInversionDM(DM dmConductivity, const Grid *grid,
                                  DM *dmInv);

/**
 * @brief Creates a KSP for inversion (same setup as solveCsemSystem, no solve).
 *
 * `Gbddc` is the forward-formulation topological discrete-gradient operator
 * passed to PCBDDC (distinct from the inversion gradient ∂F/∂X). The caller
 * must KSPDestroy it after the forward + adjoint solves.
 *
 * @param[in]  iparams  Inversion parameters.
 * @param[in]  dm       H(curl) DM.
 * @param[in]  A        System matrix.
 * @param[in]  Gbddc    Topological discrete-gradient operator for PCBDDC.
 * @param[out] ksp      Created KSP bound to A.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode createInvKSP(const imParams *iparams,
                             const DM dm,
                             const Mat A,
                             const Mat Gbddc,
                             KSP *ksp);

/**
 * @brief Solves A·sol = rhs with an already-created (factored) KSP.
 *
 * @param[in]  ksp  KSP previously created by createInvKSP.
 * @param[in]  rhs  Right-hand side vector.
 * @param[out] sol  Solution vector.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode solveInvSystem(const KSP ksp, const Vec rhs, Vec sol);

/**
 * @brief Objective + gradient callback type for L-BFGS.
 *
 * Signature: (Vec X, PetscReal *F, Vec G, void *ctx). Must be called
 * collectively on all MPI ranks.
 */
typedef PetscErrorCode (*InversionObjGradFn)(Vec X, PetscReal *F, Vec G,
                                             void *ctx);

/**
 * @brief Objective/gradient callback implementation for the inverse kernel.
 *
 * @param[in]  X    Current log-perturbation iterate.
 * @param[out] F    Objective value at X.
 * @param[out] G    Gradient at X.
 * @param[in]  ctx  InversionContext pointer.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode inversionObjGrad(Vec X, PetscReal *F, Vec G, void *ctx);

/**
 * @brief L-BFGS optimizer (Nocedal 1980, two-loop recursion).
 *
 * Replaces PETSc TAO, which is unavailable with complex scalars. If rmsPtr is
 * non-NULL and rmsTol > 0, the optimizer exits as soon as *rmsPtr <= rmsTol
 * after an accepted step - matches MATLAB's `rms <= 1.05` exit in Ex_inv.m.
 * The callback is expected to update *rmsPtr before returning.
 *
 * @param[in]     objgrad    Objective/gradient callback.
 * @param[in]     ctx        Opaque context passed to objgrad.
 * @param[in,out] X          Initial iterate; final iterate on return.
 * @param[in]     M          L-BFGS memory size.
 * @param[in]     maxIter    Maximum number of iterations.
 * @param[in]     gtol       Gradient-norm convergence tolerance.
 * @param[in]     rmsPtr     Optional pointer to a caller-updated RMS value.
 * @param[in]     rmsTol     RMS early-stop threshold (≤0 disables).
 * @param[out]    numIters   Number of iterations performed.
 * @param[out]    reasonStr  Human-readable convergence/stop reason.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode lbfgsOptimize(InversionObjGradFn objgrad, void *ctx,
                             Vec X, PetscInt M, PetscInt maxIter,
                             PetscReal gtol,
                             const PetscReal *rmsPtr, PetscReal rmsTol,
                             PetscInt *numIters, const char **reasonStr);

/**
 * @brief Writes a VTU snapshot of the current conductivity model (ρ = 1/σ).
 *
 * Called after accepted L-BFGS steps when snapshotInterval > 0; the output
 * file is {output_dir}/inv_model_iter{N:05d}.vtu.
 *
 * @param[in] ctx           Inversion context.
 * @param[in] acceptedIter  Index of the accepted iteration (used in filename).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode writeInversionSnapshotVTU(const InversionContext *ctx,
                                          PetscInt                acceptedIter);

/**
 * @brief Top-level inversion driver.
 *
 * `receivers` is the serial Vec (PETSC_COMM_SELF, length 3·N_recv) loaded by
 * loadCsemInputs from /receivers in the unified PETGEM input HDF5. It is
 * consumed once by buildReceiverInterpolationMatrices; the caller retains
 * ownership for cleanup.
 *
 * @param[in]  iparams       Inversion parameters.
 * @param[in]  dm            H(curl) DM.
 * @param[in]  grid          Finite-element grid descriptor.
 * @param[in]  conductivity  Initial per-cell conductivity Vec.
 * @param[in]  materialsID   Per-cell material-id Vec.
 * @param[in]  receivers     Serial Vec of 3·N_recv receiver coordinates.
 * @param[out] tAssemblyOut  Accumulated assembly time (seconds) for reporting.
 * @param[out] tSolverOut    Accumulated solver time (seconds) for reporting.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode runCsemInversion(const imParams  *iparams,
                                const DM          dm,
                                const Grid       *grid,
                                Vec               conductivity,
                                Vec               materialsID,
                                Vec               receivers,
                                PetscLogDouble   *tAssemblyOut,
                                PetscLogDouble   *tSolverOut);

/**
 * @brief Frees the NeighborGraph memory.
 *
 * @param[in,out] graph  Neighbor graph to destroy.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode destroyNeighborGraph(NeighborGraph *graph);

#endif /* INVERSION_H */
