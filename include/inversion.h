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
#include "receiver_interp.h"
#include "transmitter.h"
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
/* ------------------------------------------------------------------ */
/* Inversion source: extends CsemSource with per-source frequency     */
/* ------------------------------------------------------------------ */
typedef struct {
  PetscReal freq;           /* Frequency (Hz)            */
  PetscReal position[3];    /* Transmitter position (x,y,z) */
  PetscReal current;        /* Electric current          */
  PetscReal length;         /* Dipole length             */
  PetscReal dipAngle;       /* Dip angle                 */
  PetscReal azimuthAngle;   /* Azimuth angle             */
} InvCsemSource;

/* ------------------------------------------------------------------ */
/* Inversion parameters (read from PETSc options database)            */
/* ------------------------------------------------------------------ */
typedef struct {
  /* FEM parameters (shared with forward kernel) */
  PetscInt  nord;                                  /* basis order (1..6)     */

  /* Inversion-specific parameters */
  /* Path to the unified bundle (mirror of csemParams.inputFile, stashed
   * here so runCsemInversion can read /observed/Ex from the bundle
   * without taking csemParams as an arg). Set by im.csem at startup. */
  char      bundleFile[PETSC_MAX_PATH_LEN];
  PetscInt  maxIter;                               /* max L-BFGS iterations  */
  PetscInt  lbfgsMemory;                           /* L-BFGS M parameter     */
  PetscReal lambda;                                /* Tikhonov factor        */
  PetscReal errorLevel;                            /* relative data error    */
  PetscReal gtol;                                  /* gradient convergence   */
  PetscReal rmsTol;                                /* RMS early-stop (<=0 off)*/
  PetscReal diagGradientWeight;                    /* self-weight in smoother*/

  /* Fixed material IDs: cells whose material_id matches one of these values
   * are excluded from gradient smoothing (treated as self-referencing).
   * Defaults come from the bundle's /inv_meta/fixed_materials dataset
   * (written by the preprocess from sigmas.csv's `fixed` column);
   * -inv_fixed_materials on the CLI is an override.                       */
  PetscInt  numFixedMaterials;                     /* number of fixed IDs    */
  PetscInt  fixedMaterials[INV_MAX_FIXED_MATERIALS]; /* fixed material ID list */

  /* Provenance flags: PETSC_TRUE iff the corresponding field was set on the
   * CLI (and so should NOT be overridden by the bundle reader).            */
  PetscBool errorLevelFromCLI;
  PetscBool fixedMaterialsFromCLI;

  /* Source-frequency pairs loaded from the unified bundle's /inv_sources
   * group (freq, position, current, length, dipAngle, azimuthAngle).
   * Populated by setupInversionSources.                                */
  PetscInt       numFreqs;                         /* number of entries      */
  PetscReal      allFreqs[INV_MAX_FREQUENCIES];    /* frequency per entry    */
  InvCsemSource  invSources[INV_MAX_FREQUENCIES];  /* source per entry       */

  /* VTU snapshot: write conductivity model every N accepted L-BFGS steps.
   * 0 (default) disables snapshots.  Output dir is taken from -output_dir. */
  PetscInt  snapshotInterval;                      /* 0 = disabled           */

  /* Developer-only finite-difference gradient check.  When > 0, runs a
   * one-shot FD sweep on this many non-fixed cells at iter 0, prints
   * ratios (FD approx / adjoint G[i]), then skips L-BFGS.  Production
   * runs leave this at 0; enable via -inv_dev_fd_check N. */
  PetscInt  fdCheckCells;                          /* 0 = disabled           */
} invParams;

/* ------------------------------------------------------------------ */
/* Neighbor smoothing graph (CSR format)                               */
/* Fixed elements: material_id matches any entry in fixedMaterials     */
/* ------------------------------------------------------------------ */
typedef struct {
  PetscInt   numLocalCells;
  PetscInt  *neighborStart;    /* CSR row pointer, size numLocalCells+1  */
  PetscInt  *neighborList;     /* local cell IDs of neighbors            */
  PetscReal *neighborWeights;  /* normalized 1/dist weights              */
  PetscBool *isFixed;          /* PETSC_TRUE for fixed-material cells    */

  /* Fully-parallel block-Jacobi GS state, set by setupParallelSmoothingGraph.
   * When hasParallelGraph is true (production path for MPI > 1),
   * applyGaussSeidelSmoothing runs forward + reverse Gauss-Seidel ON
   * EACH RANK's owned cells using its own local copy plus one layer of
   * ghost values, exchanging ghosts between sweeps (DMLocalToGlobal /
   * DMGlobalToLocal pair). Cost is O(local cells) per smoother call,
   * no rank-0 bottleneck. NULL/false on single-rank runs (the local
   * sweep in applyGaussSeidelSmoothing handles serial directly). */
  PetscBool  hasParallelGraph;
  DM         dmInversionOver;   /* overlap=1 1-DOF/cell DM (owned)      */
  PetscInt  *oNeighborStart;    /* CSR row ptr [numLocalCells+1]        */
  PetscInt  *oNeighborList;     /* local-overlap-1 indices              */
  PetscReal *oNeighborWeights;  /* normalized 1/dist weights            */
  Vec        oLocalScratch;     /* local Vec on dmInversionOver         */
  Vec        oGlobalScratch;    /* global Vec on dmInversionOver        */
} NeighborGraph;

/* ------------------------------------------------------------------ */
/* Context passed to the L-BFGS objective/gradient callback           */
/* ------------------------------------------------------------------ */
typedef struct {
  const invParams                *iparams;
  DM                              dm;
  DM                              dmConductivity; /* DM for conductivity (3 DOF/cell) */
  DM                              dmInversion;    /* DM for inversion  (1 DOF/cell) */
  Grid                            grid;
  Vec                             conductivity; /* current sigma model (local) */
  Vec                             X0;          /* initial log(rho) (global, 1 DOF/cell) */
  Vec                             DfDm;        /* gradient workspace (local, 1 DOF/cell) */
  Mat                             dObs;        /* observed Ex, Nfreq x Nrec dense */
  Vec                             Wweights;    /* data weights, Nfreq*Nrec */
  const ReceiverInterpolationMatrices *Q;
  const NeighborGraph            *graph;
  Vec                             notFixedMaskGlobal; /* 0 at fixed cells, 1 elsewhere (global, dmInversion) */
  Vec                             notFixedMaskLocal;  /* same as above but local (dmInversion)              */
  /* Diagnostic snapshots captured during inversionObjGrad for VTU output
   * (MATLAB parity: dfdm0, DfDM, X_guangguaqian, X_guanghuahou). NULL if
   * snapshotInterval == 0. All four live on dmInversion (1 DOF/cell). */
  Vec                             DfDmRaw;     /* DfDm after chain rule, before fixed-zero and smoothing (local) */
  Vec                             DfDmFinal;   /* gradient after smoothing + regularization + global fixed-zero (global) */
  Vec                             XPreSmooth;  /* X as passed to objgrad (global copy of input X)               */
  Vec                             XPostSmooth; /* X after Gauss-Seidel smoothing inside applyLogToSigma (local) */
  PetscReal                      *allRMS;      /* output: RMS per iter   */
  PetscReal                       lastRMS;     /* RMS at most-recent callback call */
  PetscReal                       lastDataMisfit; /* data term at most-recent eval */
  PetscReal                       lastRegTerm;    /* Tikhonov term at most-recent eval */
  PetscInt                        iterCount;   /* total objgrad calls     */
  PetscInt                        acceptedIter;/* accepted L-BFGS steps  */
  /* Diagnostic flag: when PETSC_TRUE, inversionObjGrad skips BOTH the
   * forward smoother (on X inside applyLogToSigma) AND the gradient
   * smoother. Used by runFdGradientCheck so the FD test evaluates a
   * self-consistent objective F(sigma(X,X0)) vs its true adjoint gradient. */
  PetscBool                       bypassSmoother;

  /* ---- Pre-allocated workspace ----
   * Owned by setupInversionWorkspace; freed by destroyInversionWorkspace.
   * All fields below are populated once before the L-BFGS loop and reused
   * across every objgrad evaluation. Their contents do not depend on the
   * iterate X, so precomputing yields byte-identical numerical results to
   * recomputing on every callback. */

  /* Per-cell elemental-matrix work buffers used by computeGradientContribution
   * (numDofInCell × numDofInCell each, zeroed per cell inside the loop). */
  PetscReal                      *MeBuf;        /* contiguous backing (numDof²) */
  PetscReal                      *KeBuf;        /* contiguous backing (numDof²) */
  PetscReal                     **MeRows;       /* row-of-pointers view of MeBuf */
  PetscReal                     **KeRows;       /* row-of-pointers view of KeBuf */
  /* 3D quadrature for elemental mass-matrix integration. Depends only on
   * iparams->nord (constant across the run). */
  Quadrature3D                    quad3d;
  PetscBool                       quad3dInited; /* PETSC_TRUE once quad3d is filled */

  /* Per-iteration reusable Vecs (global, sized on dm). DMCreateGlobalVector
   * caches storage on the DM, but moving them out of the callback still
   * saves the 4× DMCreateGlobalVector + 2× VecCreate calls per iter and
   * keeps the same buffers warm across iterations. */
  Vec                             bVec;        /* RHS workspace (forward solve)         */
  Vec                             xVec;        /* forward solution                      */
  Vec                             nBvec;       /* adjoint RHS                           */
  Vec                             nxVec;       /* adjoint solution                      */
  Vec                             ExRecvVec;   /* Ex at receivers, parallel             */
  Vec                             wcdtDvec;    /* weighted residual workspace, parallel */

  /* Per-frequency precomputed inputs. The RHS Vec, weights Vec, and
   * observed-Ex row depend only on the source, the observed data, and
   * iparams->errorLevel — all constant across L-BFGS iterations. */
  Vec                            *Bvec_per_freq;     /* numFreqs, sized on dm   */
  Vec                            *Wf_per_freq;       /* numFreqs, sized seq Nrec */
  Vec                            *dObsRow_per_freq;  /* numFreqs, sized seq Nrec */
  PetscInt                        numFreqsAlloc;     /* size of the arrays above */

  /* Cached LHS matrices.  K (stiffness, σ-independent) and G_BDDC
   * (topological vertex incidence, σ-independent) are built ONCE at
   * setup via assembleCsemKandM and reused across all L-BFGS iters.
   * Ms keeps the sparsity from that build but its values are refilled
   * every callback via assembleCsemMsRefill against the current σ. */
  Mat                             Kmat;
  Mat                             Msmat;
  Mat                             Gmat_BDDC;
} InversionContext;

/* ------------------------------------------------------------------ */
/* Function prototypes                                                 */
/* ------------------------------------------------------------------ */

/* Read inversion parameters from PETSc options database */
PetscErrorCode readInversionParams(invParams *iparams);

/* Load case-property defaults from the bundle (error_level attribute on
 * /observed, fixed_materials array under /inv_meta) and apply to iparams
 * UNLESS the corresponding CLI override was present (see the
 * *FromCLI provenance flags above).  Safe to call even when the bundle
 * has no such entries — iparams just keeps the readInversionParams
 * defaults. */
PetscErrorCode loadInversionMetaFromBundle(const char *bundleFile,
                                            invParams  *iparams);

/* Load multi-frequency inversion sources from the unified bundle's
 * /inv_sources group (replaces the legacy text-file format).
 * `bundleFile` is the same HDF5 path consumed by loadCsemInputs.
 * Populates iparams->numFreqs, allFreqs[], invSources[]. */
PetscErrorCode setupInversionSources(const char *bundleFile,
                                     invParams  *iparams);

/* Load observed data from the unified bundle's /observed/Ex dataset
 * (HDF5 compound complex128, shape [numFreqs, numReceivers]).
 * Returns dense Mat of size numFreqs x numReceivers on PETSC_COMM_SELF. */
PetscErrorCode loadObservedData(const char *bundleFile,
                                PetscInt    numFreqs,
                                PetscInt    numReceivers,
                                Mat        *dObs);

/* Build CSR neighbor smoothing graph from DMPlex topology.
 * Cells whose material_id matches any entry in iparams->fixedMaterials
 * are treated as fixed (self-referencing only, excluded from smoothing). */
PetscErrorCode buildNeighborSmoothingGraph(const DM dm,
                                           const Grid *grid,
                                           const invParams *iparams,
                                           Vec materialsID,
                                           NeighborGraph *graph);

/* Accumulate per-element adjoint gradient into DfDm (1 DOF/cell).
 * DfDm[ie] += real( (-2*constFactor*Me_e*x_e)^T · nx_e )
 * Plain transpose (no conjugate), matching MATLAB iG.'*inx.
 * `quadrature_3d`, `Me`, `Ke` are workspace buffers owned by the caller
 * (set up once on InversionContext via setupInversionWorkspace). */
PetscErrorCode computeGradientContribution(const invParams *iparams,
                                           const DM dm,
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

/* Apply forward + reverse Gauss-Seidel smoothing to vector v.
 * Gathers to rank 0, sweeps, scatters back. */
PetscErrorCode applyGaussSeidelSmoothing(const NeighborGraph *graph,
                                         PetscReal diagWeight,
                                         Vec v);

/* Recover conductivity from log-conductivity perturbation:
 * sigma = 1 / exp(X_smooth + X0)  (isotropic, written to components 0-2).
 * X and X0 are global Vecs on dmInversion (1 DOF/cell).
 * sigmaModel is a local Vec on dmConductivity (4 DOF/cell).
 * graph/diagWeight: GS smoothing applied to a local copy of X before exp,
 * matching MATLAB's tempX smoothing after each L-BFGS step.
 * xPostSmoothOut (optional, may be NULL): if provided, must be a local Vec
 * on dmInversion; receives the smoothed X (MATLAB's tempX post-smoothing,
 * pre-X0 add) for VTU diagnostic snapshots. */
PetscErrorCode applyLogToSigma(DM dmInversion, DM dmConductivity,
                                const Vec X, const Vec X0,
                                Vec sigmaModel, const Grid *grid,
                                const NeighborGraph *graph,
                                PetscReal diagWeight,
                                Vec xPostSmoothOut);

/* Create a DM with 1 DOF per cell for the inversion variables (X, X0,
 * gradient).  Clones the DMPlex topology from dmConductivity but sets a
 * new section with a single scalar DOF per cell. */
PetscErrorCode createInversionDM(DM dmConductivity, const Grid *grid,
                                  DM *dmInv);

/* Create KSP for inversion (same setup as solveCsemSystem but no solve).
 * Caller must KSPDestroy after forward + adjoint solves. */
PetscErrorCode createInvKSP(const invParams *iparams,
                             const DM dm,
                             const Mat A,
                             const Mat G,
                             KSP *ksp);

/* Solve A*sol = rhs with an already-created (factored) KSP */
PetscErrorCode solveInvSystem(const KSP ksp, const Vec rhs, Vec sol);

/* Objective + gradient callback for L-BFGS.
 * Signature: (Vec X, PetscReal *F, Vec G, void *ctx).
 * Must be called collectively on all MPI ranks. */
typedef PetscErrorCode (*InversionObjGradFn)(Vec X, PetscReal *F, Vec G,
                                             void *ctx);

/* Callback implementation for the CSEM inverse kernel */
PetscErrorCode inversionObjGrad(Vec X, PetscReal *F, Vec G, void *ctx);

/* L-BFGS optimizer (Nocedal 1980, two-loop recursion).
 * Replaces PETSc TAO which is unavailable with complex scalars.
 *
 * If rmsPtr != NULL and rmsTol > 0, the optimizer exits as soon as
 * *rmsPtr <= rmsTol after an accepted step — matches MATLAB's
 * `rms <= 1.05` exit in Ex_inv.m. The callback is expected to update
 * *rmsPtr before returning. */
PetscErrorCode lbfgsOptimize(InversionObjGradFn objgrad, void *ctx,
                             Vec X, PetscInt M, PetscInt maxIter,
                             PetscReal gtol,
                             const PetscReal *rmsPtr, PetscReal rmsTol,
                             PetscInt *numIters, const char **reasonStr);

/* Write VTU snapshot of the current conductivity model (rho = 1/sigma).
 * Called after accepted L-BFGS steps when snapshotInterval > 0.
 * Output file: {output_dir}/inv_model_iter{N:05d}.vtu */
PetscErrorCode writeInversionSnapshotVTU(const InversionContext *ctx,
                                          PetscInt                acceptedIter);

/* Top-level inversion driver.
 *
 * `receivers` is the serial Vec (PETSC_COMM_SELF, length 3·N_recv) loaded
 * by loadCsemInputs from /receivers in the unified PETGEM input HDF5.
 * It is consumed once by buildReceiverInterpolationMatrices and the
 * caller retains ownership for cleanup. */
PetscErrorCode runCsemInversion(const invParams  *iparams,
                                const DM          dm,
                                const Grid       *grid,
                                Vec               conductivity,
                                Vec               materialsID,
                                Vec               receivers);

/* Free NeighborGraph memory */
PetscErrorCode destroyNeighborGraph(NeighborGraph *graph);

#endif /* INVERSION_H */
