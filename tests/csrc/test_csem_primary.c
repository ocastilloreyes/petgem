/*
 * Filename: test_csem_primary.c
 * Author: PETGEM test suite
 * Date: 2026-11-20
 *
 * Description:
 * Primary field and secondary right-hand side (csem_primary.c) on a box mesh
 * with two layers and a conductive block:
 *   - createCsemPrimary: background conductivity per cell, cells with
 *     sigma != sigma_b and cells across a layer interface (layered and model);
 *     material ids outside the background table rejected;
 *   - assembleCsemSecondaryRhs equals iωμ Ms(σ - σ_b) E_p,h for two source
 *     sets at different frequencies, Ms assembled by assembleMaxwellOperator
 *     (layered background, natural boundary);
 *   - model background: x_s + E_p,h = x_t with MUMPS, at the DOFs and at the
 *     receivers (addCsemPrimaryAtReceivers); a background equal to the model
 *     gives b = 0;
 *   - layered background: addCsemPrimaryAtReceivers adds the analytic field;
 *     the interpolated E_p,h and curl(E_p,h)/(iωμ) match the analytic E and H
 *     in a small box away from the source (the curl converges one order
 *     below E), and shared DOFs hold the same value on every rank; a source
 *     in a cell with sigma != sigma_b is rejected;
 *   - receivers on a layer interface take the layer of the cell that
 *     interpolates them (Q->cellCentroids): addCsemPrimaryAtReceivers and the
 *     interpolated E_p,h match the analytic E of that layer.
 *
 * Usage: test_csem_primary <order>   (order 1-4) Exit 0 iff all checks pass.
 */
/* PETSc libraries */
#include <petsc.h>

/* PETGEM functions */
#include "assembly.h"
#include "box_mesh.h"
#include "constants.h"
#include "csem.h"
#include "csem_layered.h"
#include "csem_primary.h"
#include "grid.h"
#include "petgem_test.h"
#include "receiver_interp.h"
#include "solver.h"

static const PetscInt  boxFaces[3]       = {4, 4, 4};
static const PetscReal boxLower[3]       = {-600.0, -600.0, -1000.0};
static const PetscReal boxUpper[3]       = {600.0, 600.0, 0.0};
static const PetscReal materialSigma[9]  = {0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0};
static const PetscReal backgroundSigma[] = {0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static const PetscReal freq              = 1.0;
static const PetscReal receiverXyz[9]    = {100.0, 50.0, -10.0, -400.0, 300.0, -600.0, 250.0, -350.0, -900.0};

/**
 * @brief Material at a point.
 *
 * Material 0 above z = -500, 1 below, 2 in the block (-300, 0) x (0, 300) x
 * (-750, -500).
 *
 * @param[in] x  Point coordinates.
 *
 * @return PetscInt Material id.
 */
static PetscInt materialAt(const PetscReal x[3]) {
  if (x[0] > -300.0 && x[0] < 0.0 && x[1] > 0.0 && x[1] < 300.0 && x[2] > -750.0 && x[2] < -500.0) return 2;
  return x[2] > -500.0 ? 0 : 1;
}

/**
 * @brief Material id at a point as a real (cell field callback).
 *
 * @param[in] x  Point coordinates.
 *
 * @return PetscReal Material id.
 */
static PetscReal materialOf(const PetscReal x[3]) {
  return (PetscReal)materialAt(x);
}

/**
 * @brief Conductivity of the material at a point (cell field callback).
 *
 * @param[in] x  Point coordinates.
 *
 * @return PetscReal sigma_x of the material.
 */
static PetscReal sigmaOf(const PetscReal x[3]) {
  return materialSigma[3 * materialAt(x)];
}

/**
 * @brief Per-cell material id Vec laid out as loadModelInputs() builds it.
 *
 * @param[in]  dm   DMPlex mesh.
 * @param[out] mat  Per-cell material id.
 */
static void create_cell_material(DM dm, Vec *mat) {
  DM                 dmM;
  PetscSection       sec;
  PetscInt           pStart, pEnd, cStart, cEnd;
  Vec                tmp;
  PetscScalar       *a;
  const PetscScalar *s;

  /* Three equal components from create_cell_conductivity, first one kept */
  create_cell_conductivity(dm, materialOf, &tmp);
  PetscCallAbort(PETSC_COMM_WORLD, DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCallAbort(PETSC_COMM_WORLD, DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCallAbort(PETSC_COMM_WORLD, DMClone(dm, &dmM));
  PetscCallAbort(PETSC_COMM_WORLD, PetscSectionCreate(PETSC_COMM_WORLD, &sec));
  PetscCallAbort(PETSC_COMM_WORLD, PetscSectionSetChart(sec, pStart, pEnd));
  for (PetscInt c = cStart; c < cEnd; c++) PetscCallAbort(PETSC_COMM_WORLD, PetscSectionSetDof(sec, c, 1));
  PetscCallAbort(PETSC_COMM_WORLD, PetscSectionSetUp(sec));
  PetscCallAbort(PETSC_COMM_WORLD, DMSetLocalSection(dmM, sec));
  PetscCallAbort(PETSC_COMM_WORLD, PetscSectionDestroy(&sec));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(dmM, mat));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(*mat, &a));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(tmp, &s));
  for (PetscInt c = cStart; c < cEnd; c++) a[c - cStart] = s[3 * (c - cStart)];
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(tmp, &s));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(*mat, &a));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&tmp));
  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&dmM));
}

/**
 * @brief Box mesh with the H(curl) space of one order, its conductivity and
 *        material ids.
 *
 * @param[in]  order  Basis order.
 * @param[in]  bc     Boundary condition.
 * @param[out] dm     Box mesh.
 * @param[out] grid   Grid.
 * @param[out] sigma  Per-cell conductivity.
 * @param[out] mat    Per-cell material id.
 */
static void setup(PetscInt order, PetgemBoundaryCondition bc, DM *dm, Grid *grid, Vec *sigma, Vec *mat) {
  create_box_mesh(boxFaces, boxLower, boxUpper, 0.0, dm);
  PetscCallAbort(PETSC_COMM_WORLD, setupNedelecGrid(order, bc, dm, grid));
  create_cell_conductivity(*dm, sigmaOf, sigma);
  create_cell_material(*dm, mat);
}

/**
 * @brief Destroys the objects created by setup().
 *
 * @param[in,out] dm     Box mesh.
 * @param[in,out] grid   Grid.
 * @param[in,out] sigma  Per-cell conductivity.
 * @param[in,out] mat    Per-cell material id.
 */
static void teardown(DM *dm, Grid *grid, Vec *sigma, Vec *mat) {
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(mat));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(sigma));
  PetscCallAbort(PETSC_COMM_WORLD, destroyGrid(grid));
  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(dm));
}

/* Two transmitters in the top layer, frequency freq. */
static CsemSource sourceArray[2] = {{{150.0, -150.0, -200.0}, 2.0, 50.0, 20.0, 30.0}, {{-140.0, 160.0, -230.0}, 1.0, 100.0, -10.0, 120.0}};

/**
 * @brief Two transmitters in the top layer at frequency freq.
 *
 * @return CsemSourceSet Source set.
 */
static CsemSourceSet sourceSet(void) {
  CsemSourceSet set = {freq, 2, sourceArray};
  return set;
}

/**
 * @brief Two-layer background with one interface.
 *
 * @param[in]  interface  Height of the interface (m).
 * @param[out] bg         Background model.
 */
static void layered_background(PetscReal interface, CsemBackground *bg) {
  const PetscReal s[2] = {0.1, 1.0};
  PetscCallAbort(PETSC_COMM_WORLD, PetscMemzero(bg, sizeof(*bg)));
  bg->type = CSEM_PRIMARY_LAYERED;
  PetscCallAbort(PETSC_COMM_WORLD, createLayeredModel(2, &interface, s, s, &bg->layered));
}

/**
 * @brief Model background with numMaterials rows of a conductivity table.
 *
 * @param[in]  table         sigma_x, sigma_y, sigma_z per material.
 * @param[in]  numMaterials  Number of materials.
 * @param[out] bg            Background model.
 */
static void model_background(const PetscReal *table, PetscInt numMaterials, CsemBackground *bg) {
  PetscCallAbort(PETSC_COMM_WORLD, PetscMemzero(bg, sizeof(*bg)));
  bg->type         = CSEM_PRIMARY_MODEL;
  bg->numMaterials = numMaterials;
  PetscCallAbort(PETSC_COMM_WORLD, PetscMalloc1(3 * numMaterials, &bg->materialSigma));
  PetscCallAbort(PETSC_COMM_WORLD, PetscArraycpy(bg->materialSigma, table, 3 * numMaterials));
}

/**
 * @brief Global Vec of E_p,h of source k of set 0.
 *
 * @param[in]  dm       DMPlex carrying the H(curl) section.
 * @param[in]  primary  Primary field.
 * @param[in]  k        Source.
 * @param[out] x        Global Vec.
 */
static void primary_global(const DM dm, const CsemPrimary *primary, PetscInt k, Vec *x) {
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(dm, x));
  PetscCallAbort(PETSC_COMM_WORLD, DMLocalToGlobal(dm, primary->Ep[0][k], INSERT_VALUES, *x));
}

/**
 * @brief Sequential receiver Vec of n points.
 *
 * @param[in]  n    Number of points.
 * @param[in]  xyz  Point coordinates.
 * @param[out] r    Receiver Vec.
 */
static void receiver_vec(PetscInt n, const PetscReal *xyz, Vec *r) {
  PetscScalar *a;
  PetscCallAbort(PETSC_COMM_SELF, VecCreateSeq(PETSC_COMM_SELF, 3 * n, r));
  PetscCallAbort(PETSC_COMM_SELF, VecSetBlockSize(*r, 3));
  PetscCallAbort(PETSC_COMM_SELF, VecGetArray(*r, &a));
  for (PetscInt i = 0; i < 3 * n; i++) a[i] = xyz[i];
  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(*r, &a));
}

/**
 * @brief Relative error max_f ||a_f - b_f|| / max_f ||b_f|| over the receiver
 *        fields.
 *
 * @param[in] a  Computed fields.
 * @param[in] b  Reference fields.
 *
 * @return PetscReal Relative error.
 */
static PetscReal fields_error(Vec a[NUM_RECEIVER_FIELDS], Vec b[NUM_RECEIVER_FIELDS]) {
  PetscReal err = 0.0, scale = 0.0;
  for (PetscInt f = 0; f < NUM_RECEIVER_FIELDS; f++) {
    Vec       d;
    PetscReal nd, nb;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(a[f], &d));
    PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(d, -1.0, b[f], a[f]));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(d, NORM_INFINITY, &nd));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(b[f], NORM_INFINITY, &nb));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&d));
    err   = PetscMax(err, nd);
    scale = PetscMax(scale, nb);
  }
  return err / scale;
}

/**
 * @brief Checks the background conductivity and the cell counts of the layered
 *        and model backgrounds.
 *
 * @param[in] order  Basis order.
 */
static void check_background(PetscInt order) {
  DM             dm;
  Grid           grid;
  Vec            sigma, mat;
  CsemBackground bg;
  CsemPrimary    primary;
  PetscErrorCode ierr;
  PetscInt       wrong = 0;

  setup(order, PETGEM_BC_PEC, &dm, &grid, &sigma, &mat);

  /* Conforming layers: sigma_b of every cell outside the block equals sigma */
  layered_background(-500.0, &bg);
  PetscCallAbort(PETSC_COMM_WORLD, createCsemPrimary(&bg, dm, &grid, sigma, NULL, &primary));
  PT_CHECK(primary.numAnomalyCells == 6, "layered: %d cells with sigma != sigma_b, expected 6", (int)primary.numAnomalyCells);
  PT_CHECK(primary.numCrossingCells == 0, "layered: %d cells across the interface, expected 0", (int)primary.numCrossingCells);
  {
    const PetscScalar *s, *sb, *m;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(sigma, &s));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(primary.sigmaBackground, &sb));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(mat, &m));
    for (PetscInt c = 0; c < grid.cellEnd - grid.cellStart; c++) {
      const PetscReal expected = (PetscRealPart(m[c]) == 2.0) ? 1.0 : PetscRealPart(s[3 * c]);
      for (PetscInt d = 0; d < 3; d++) wrong += (PetscRealPart(sb[3 * c + d]) != expected);
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(mat, &m));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(primary.sigmaBackground, &sb));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(sigma, &s));
  }
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &wrong, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PT_CHECK(wrong == 0, "layered: %d wrong background conductivity entries", (int)wrong);
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemBackground(&bg));

  /* Interface at z = -400 crosses the 16 hexahedra of -500 < z < -250 */
  layered_background(-400.0, &bg);
  PetscCallAbort(PETSC_COMM_WORLD, createCsemPrimary(&bg, dm, &grid, sigma, NULL, &primary));
  PT_CHECK(primary.numCrossingCells == 96, "layered: %d cells across z = -400, expected 96", (int)primary.numCrossingCells);
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemBackground(&bg));

  /* Model background */
  model_background(backgroundSigma, 3, &bg);
  PetscCallAbort(PETSC_COMM_WORLD, createCsemPrimary(&bg, dm, &grid, sigma, mat, &primary));
  PT_CHECK(primary.numAnomalyCells == 6, "model: %d cells with sigma != sigma_b, expected 6", (int)primary.numAnomalyCells);
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemBackground(&bg));

  /* Material 2 outside a two-row table */
  model_background(backgroundSigma, 2, &bg);
  PetscCallAbort(PETSC_COMM_WORLD, PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = createCsemPrimary(&bg, dm, &grid, sigma, mat, &primary);
  PetscCallAbort(PETSC_COMM_WORLD, PetscPopErrorHandler());
  {
    int failed = (ierr != PETSC_SUCCESS), any;
    PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_LOR, PETSC_COMM_WORLD));
    PT_CHECK(any, "material id outside the background table not rejected");
  }
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemBackground(&bg));

  teardown(&dm, &grid, &sigma, &mat);
}

/**
 * @brief Checks the secondary right-hand side b = iωμ Ms(Δσ) x_p.
 *
 * @param[in] order  Basis order.
 */
static void check_rhs(PetscInt order) {
  CsemSourceSet  sets[2] = {sourceSet(), {3.0, 1, &sourceArray[1]}};
  DM             dm;
  Grid           grid;
  Vec            sigma, mat, dsigma, b, xp, bref;
  Mat            B[2], K, Ms;
  CsemBackground bg;
  CsemPrimary    primary;

  setup(order, PETGEM_BC_NATURAL, &dm, &grid, &sigma, &mat);
  layered_background(-500.0, &bg);
  PetscCallAbort(PETSC_COMM_WORLD, createCsemPrimary(&bg, dm, &grid, sigma, NULL, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, computeCsemPrimary(2, sets, dm, &grid, NULL, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, createRhsVector(dm, &b));
  for (PetscInt s = 0; s < 2; s++) PetscCallAbort(PETSC_COMM_WORLD, createRhsMatrix(b, sets[s].numSources, &B[s]));
  PetscCallAbort(PETSC_COMM_WORLD, assembleCsemSecondaryRhs(&primary, dm, &grid, sigma, B));
  PetscCallAbort(PETSC_COMM_WORLD, logCsemFormulation(PETSC_COMM_WORLD, &primary));

  /* Reference: iωμ_s Ms(σ - σ_b) x_p */
  {
    DM dmSigma;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetDM(sigma, &dmSigma));
    PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(dmSigma, &dsigma));
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(dsigma, -1.0, primary.sigmaBackground, sigma));
  PetscCallAbort(PETSC_COMM_WORLD, assembleMaxwellOperator(dm, &grid, dsigma, 0.0, &K, &Ms, NULL));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(b, &bref));
  for (PetscInt s = 0; s < 2; s++)
    for (PetscInt k = 0; k < sets[s].numSources; k++) {
      Vec       bcol;
      PetscReal nref, nd;
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(dm, &xp));
      PetscCallAbort(PETSC_COMM_WORLD, DMLocalToGlobal(dm, primary.Ep[s][k], INSERT_VALUES, xp));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(Ms, xp, bref));
      PetscCallAbort(PETSC_COMM_WORLD, VecScale(bref, iOmegaMu(sets[s].freq)));
      PetscCallAbort(PETSC_COMM_WORLD, VecNorm(bref, NORM_2, &nref));
      PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetColumnVecRead(B[s], k, &bcol));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(bref, -1.0, bcol));
      PetscCallAbort(PETSC_COMM_WORLD, MatDenseRestoreColumnVecRead(B[s], k, &bcol));
      PetscCallAbort(PETSC_COMM_WORLD, VecNorm(bref, NORM_2, &nd));
      PT_CHECK(nref > 0.0 && nd <= 1e-12 * nref, "set %d, source %d: |b - iωμ Ms(Δσ) x_p| / |b| = %.3e (|b| = %.3e)", (int)s, (int)k,
               (double)(nd / nref), (double)nref);
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&xp));
    }

  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&bref));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&Ms));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&K));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&dsigma));
  for (PetscInt s = 0; s < 2; s++) PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&B[s]));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&b));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemBackground(&bg));
  teardown(&dm, &grid, &sigma, &mat);
}

/**
 * @brief Checks that x_s + E_p,h of the model background matches the total
 *        field.
 *
 * @param[in] order  Basis order.
 */
static void check_model_decomposition(PetscInt order) {
  const PetscScalar             iwmu = iOmegaMu(freq);
  CsemSourceSet                 set  = sourceSet();
  DM                            dm;
  Grid                          grid;
  Vec                           sigma, mat, b, receivers, ft[NUM_RECEIVER_FIELDS], fs[NUM_RECEIVER_FIELDS];
  Mat                           A, G, Bt, Bs, Xt, Xs;
  CsemBackground                bg;
  CsemPrimary                   primary;
  SolveInfo                     info;
  ReceiverInterpolationMatrices Q;

  setup(order, PETGEM_BC_PEC, &dm, &grid, &sigma, &mat);
  receiver_vec(3, receiverXyz, &receivers);
  PetscCallAbort(PETSC_COMM_WORLD, buildReceiverInterpolationMatrices(receivers, dm, &grid, &Q));
  PetscCallAbort(PETSC_COMM_WORLD, createReceiverFields(&Q, ft));
  PetscCallAbort(PETSC_COMM_WORLD, createReceiverFields(&Q, fs));

  /* Total field */
  PetscCallAbort(PETSC_COMM_WORLD, assembleCsemRhs(set, dm, &grid, iwmu, &Bt));
  PetscCallAbort(PETSC_COMM_WORLD, assembleMaxwellOperator(dm, &grid, sigma, iwmu, &A, NULL, &G));
  PetscCallAbort(PETSC_COMM_WORLD, solveMaxwellSystem(dm, A, Bt, G, order, NULL, &Xt, &info));

  /* Secondary field with the model background */
  model_background(backgroundSigma, 3, &bg);
  PetscCallAbort(PETSC_COMM_WORLD, createCsemPrimary(&bg, dm, &grid, sigma, mat, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, computeCsemPrimary(1, &set, dm, &grid, NULL, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, createRhsVector(dm, &b));
  PetscCallAbort(PETSC_COMM_WORLD, createRhsMatrix(b, set.numSources, &Bs));
  PetscCallAbort(PETSC_COMM_WORLD, assembleCsemSecondaryRhs(&primary, dm, &grid, sigma, &Bs));
  PetscCallAbort(PETSC_COMM_WORLD, solveMaxwellSystem(dm, A, Bs, G, order, NULL, &Xs, &info));
  PetscCallAbort(PETSC_COMM_WORLD, logCsemFormulation(PETSC_COMM_WORLD, &primary));

  for (PetscInt k = 0; k < set.numSources; k++) {
    Vec       xt, xs, xp;
    PetscReal nt, nd;
    primary_global(dm, &primary, k, &xp);
    PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetColumnVecRead(Xt, k, &xt));
    PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetColumnVecRead(Xs, k, &xs));
    PetscCallAbort(PETSC_COMM_WORLD, interpolateReceiverFields(&Q, xt, iwmu, ft));
    PetscCallAbort(PETSC_COMM_WORLD, interpolateReceiverFields(&Q, xs, iwmu, fs));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(xt, NORM_2, &nt));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(xp, 1.0, xs));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(xp, -1.0, xt));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(xp, NORM_2, &nd));
    PetscCallAbort(PETSC_COMM_WORLD, MatDenseRestoreColumnVecRead(Xs, k, &xs));
    PetscCallAbort(PETSC_COMM_WORLD, MatDenseRestoreColumnVecRead(Xt, k, &xt));
    PT_CHECK(nd <= 1e-8 * nt, "model, source %d: |x_s + x_p - x_t| / |x_t| = %.3e", (int)k, (double)(nd / nt));

    PetscCallAbort(PETSC_COMM_WORLD, addCsemPrimaryAtReceivers(&primary, 0, k, dm, &Q, receivers, fs));
    {
      const PetscReal e = fields_error(fs, ft);
      PT_CHECK(e <= 1e-8, "model, source %d: receiver fields of x_s + E_p differ from x_t by %.3e", (int)k, (double)e);
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&xp));
  }
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemBackground(&bg));

  /* Background equal to the model: no cell differs and b = 0 */
  model_background(materialSigma, 3, &bg);
  PetscCallAbort(PETSC_COMM_WORLD, createCsemPrimary(&bg, dm, &grid, sigma, mat, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, computeCsemPrimary(1, &set, dm, &grid, NULL, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, assembleCsemSecondaryRhs(&primary, dm, &grid, sigma, &Bs));
  {
    PetscReal nb;
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(Bs, NORM_FROBENIUS, &nb));
    PT_CHECK(primary.numAnomalyCells == 0 && nb == 0.0, "background = model: %d cells differ, |b| = %.3e", (int)primary.numAnomalyCells,
             (double)nb);
  }
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemBackground(&bg));

  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&Xs));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&Bs));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&b));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&Xt));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&G));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&A));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&Bt));
  PetscCallAbort(PETSC_COMM_WORLD, destroyReceiverFields(fs));
  PetscCallAbort(PETSC_COMM_WORLD, destroyReceiverFields(ft));
  PetscCallAbort(PETSC_COMM_WORLD, destroyReceiverInterpolationMatrices(&Q));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&receivers));
  teardown(&dm, &grid, &sigma, &mat);
}

/**
 * @brief Checks the analytic E and H of each source at the receivers, by
 *        rank-local evaluation.
 *
 * @param[in] order  Basis order.
 */
static void check_layered_receivers(PetscInt order) {
  CsemSourceSet                 set = sourceSet();
  DM                            dm;
  Grid                          grid;
  Vec                           sigma, mat, receivers, f[NUM_RECEIVER_FIELDS];
  CsemBackground                bg;
  CsemPrimary                   primary;
  ReceiverInterpolationMatrices Q;
  LayeredDipole                 dipole;
  PetscScalar                   E[9], H[9];
  PetscReal                     moment[3], direction[3], err = 0.0;
  PetscInt                      lo, hi;

  setup(order, PETGEM_BC_PEC, &dm, &grid, &sigma, &mat);
  receiver_vec(3, receiverXyz, &receivers);
  PetscCallAbort(PETSC_COMM_WORLD, buildReceiverInterpolationMatrices(receivers, dm, &grid, &Q));
  PetscCallAbort(PETSC_COMM_WORLD, createReceiverFields(&Q, f));
  for (PetscInt i = 0; i < NUM_RECEIVER_FIELDS; i++) PetscCallAbort(PETSC_COMM_WORLD, VecZeroEntries(f[i]));
  layered_background(-500.0, &bg);
  PetscCallAbort(PETSC_COMM_WORLD, createCsemPrimary(&bg, dm, &grid, sigma, NULL, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, computeCsemPrimary(1, &set, dm, &grid, NULL, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, addCsemPrimaryAtReceivers(&primary, 0, 1, dm, &Q, receivers, f));

  PetscCallAbort(PETSC_COMM_WORLD, computeVectorRotation(sourceArray[1].azimuthAngle, sourceArray[1].dipAngle, direction));
  for (PetscInt d = 0; d < 3; d++) moment[d] = direction[d] * sourceArray[1].current * sourceArray[1].length;
  PetscCallAbort(PETSC_COMM_WORLD, createLayeredDipole(&bg.layered, freq, sourceArray[1].position, moment, &dipole));
  PetscCallAbort(PETSC_COMM_WORLD, evaluateLayeredDipole(&dipole, 3, receiverXyz, NULL, E, H));
  PetscCallAbort(PETSC_COMM_WORLD, destroyLayeredDipole(&dipole));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetOwnershipRange(f[0], &lo, &hi));
  for (PetscInt i = 0; i < NUM_RECEIVER_FIELDS; i++) {
    const PetscScalar *a;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f[i], &a));
    for (PetscInt r = lo; r < hi; r++) {
      const PetscScalar ref = (i < 3) ? E[3 * r + i] : H[3 * r + i - 3];
      err                   = PetscMax(err, PetscAbsScalar(a[r - lo] - ref) / PetscAbsScalar(ref));
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f[i], &a));
  }
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
  PT_CHECK(err <= 1e-14, "layered receivers: relative difference %.3e to evaluateLayeredDipole", (double)err);

  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  PetscCallAbort(PETSC_COMM_WORLD, destroyReceiverFields(f));
  PetscCallAbort(PETSC_COMM_WORLD, destroyReceiverInterpolationMatrices(&Q));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&receivers));

  /* Source 1 in the block */
  {
    CsemSource     moved[2] = {sourceArray[0], sourceArray[1]};
    CsemSourceSet  bad      = {freq, 2, moved};
    PetscErrorCode ierr;
    moved[1].position[0] = -150.0;
    moved[1].position[1] = 150.0;
    moved[1].position[2] = -620.0;
    PetscCallAbort(PETSC_COMM_WORLD, createCsemPrimary(&bg, dm, &grid, sigma, NULL, &primary));
    PetscCallAbort(PETSC_COMM_WORLD, PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    ierr = computeCsemPrimary(1, &bad, dm, &grid, NULL, &primary);
    PetscCallAbort(PETSC_COMM_WORLD, PetscPopErrorHandler());
    PT_CHECK(ierr != PETSC_SUCCESS, "layered source in a cell with sigma != sigma_b not rejected");
    PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  }
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemBackground(&bg));
  teardown(&dm, &grid, &sigma, &mat);
}

/**
 * @brief Unit conductivity (cell field callback).
 *
 * @param[in] x  Point coordinates.
 *
 * @return PetscReal 1.
 */
static PetscReal unitSigma(const PetscReal x[3]) {
  (void)x;
  return 1.0;
}

/**
 * @brief Checks the interpolated E_p,h and curl(E_p,h)/(iωμ) against the
 *        analytic field.
 *
 * In a 40 m box 900 m from the source.
 *
 * @param[in] order  Basis order.
 */
static void check_layered_conventions(PetscInt order) {
  const PetscInt                faces[3] = {2, 2, 2};
  const PetscReal               lower[3] = {880.0, -20.0, -520.0}, upper[3] = {920.0, 20.0, -480.0};
  const PetscReal               point[3]  = {903.0, 7.0, -494.0};
  const PetscReal               tolE[4]   = {1e-1, 2e-3, 1e-5, 2e-6};
  const PetscReal               tolH[4]   = {1.5e-1, 1.5e-1, 3e-3, 5e-4};
  CsemSource                    src       = {{0.0, 0.0, -500.0}, 1.0, 1.0, 35.0, 20.0};
  CsemSourceSet                 set       = {freq, 1, &src};
  const PetscReal               sigma1[1] = {1.0};
  DM                            dm;
  Grid                          grid;
  Vec                           sigma, receivers, xp, f[NUM_RECEIVER_FIELDS];
  CsemBackground                bg;
  CsemPrimary                   primary;
  ReceiverInterpolationMatrices Q;
  LayeredDipole                 dipole;
  PetscBool                    *all;
  PetscScalar                   E[3], H[3];
  PetscReal                     moment[3], errE = 0.0, errH = 0.0, nE = 0.0, nH = 0.0;
  PetscInt                      lo, hi;

  create_box_mesh(faces, lower, upper, 0.0, &dm);
  PetscCallAbort(PETSC_COMM_WORLD, setupNedelecGrid(order, PETGEM_BC_NATURAL, &dm, &grid));
  create_cell_conductivity(dm, unitSigma, &sigma);
  PetscCallAbort(PETSC_COMM_WORLD, PetscMemzero(&bg, sizeof(bg)));
  bg.type = CSEM_PRIMARY_LAYERED;
  PetscCallAbort(PETSC_COMM_WORLD, createLayeredModel(1, NULL, sigma1, sigma1, &bg.layered));
  PetscCallAbort(PETSC_COMM_WORLD, createCsemPrimary(&bg, dm, &grid, sigma, NULL, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, PetscMalloc1(grid.cellEnd - grid.cellStart, &all));
  for (PetscInt c = 0; c < grid.cellEnd - grid.cellStart; c++) all[c] = PETSC_TRUE;
  PetscCallAbort(PETSC_COMM_WORLD, computeCsemPrimary(1, &set, dm, &grid, all, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(all));

  receiver_vec(1, point, &receivers);
  PetscCallAbort(PETSC_COMM_WORLD, buildReceiverInterpolationMatrices(receivers, dm, &grid, &Q));
  PetscCallAbort(PETSC_COMM_WORLD, createReceiverFields(&Q, f));
  primary_global(dm, &primary, 0, &xp);
  PetscCallAbort(PETSC_COMM_WORLD, interpolateReceiverFields(&Q, xp, iOmegaMu(freq), f));

  /* Shared DOFs hold the owner's value on every rank */
  {
    Vec       loc;
    PetscReal nd, ne;
    PetscCallAbort(PETSC_COMM_WORLD, DMGetLocalVector(dm, &loc));
    PetscCallAbort(PETSC_COMM_WORLD, DMGlobalToLocal(dm, xp, INSERT_VALUES, loc));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(loc, -1.0, primary.Ep[0][0]));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(loc, NORM_INFINITY, &nd));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(xp, NORM_INFINITY, &ne));
    PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &nd, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
    PetscCallAbort(PETSC_COMM_WORLD, DMRestoreLocalVector(dm, &loc));
    PT_CHECK(nd <= 1e-14 * ne, "order %d: local E_p,h differs from the owner's value by %.3e", (int)order, (double)(nd / ne));
  }

  PetscCallAbort(PETSC_COMM_WORLD, computeVectorRotation(src.azimuthAngle, src.dipAngle, moment));
  PetscCallAbort(PETSC_COMM_WORLD, createLayeredDipole(&bg.layered, freq, src.position, moment, &dipole));
  PetscCallAbort(PETSC_COMM_WORLD, evaluateLayeredDipole(&dipole, 1, point, NULL, E, H));
  PetscCallAbort(PETSC_COMM_WORLD, destroyLayeredDipole(&dipole));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetOwnershipRange(f[0], &lo, &hi));
  if (lo == 0 && hi > 0) {
    for (PetscInt d = 0; d < 3; d++) {
      const PetscScalar *a, *b;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f[RECV_EX + d], &a));
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f[RECV_HX + d], &b));
      errE = PetscMax(errE, PetscAbsScalar(a[0] - E[d]));
      errH = PetscMax(errH, PetscAbsScalar(b[0] - H[d]));
      nE   = PetscMax(nE, PetscAbsScalar(E[d]));
      nH   = PetscMax(nH, PetscAbsScalar(H[d]));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f[RECV_HX + d], &b));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f[RECV_EX + d], &a));
    }
    errE /= nE;
    errH /= nH;
  }
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &errE, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &errH, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
  PetscCallAbort(PETSC_COMM_WORLD, PetscPrintf(PETSC_COMM_WORLD, "  order %d: interpolated E error %.3e, H error %.3e\n", (int)order,
                                               (double)errE, (double)errH));
  PT_CHECK(errE <= tolE[order - 1], "order %d: interpolated E relative error %.3e > %.1e", (int)order, (double)errE,
           (double)tolE[order - 1]);
  PT_CHECK(errH <= tolH[order - 1], "order %d: curl(E_p,h)/(iωμ) relative error %.3e > %.1e", (int)order, (double)errH,
           (double)tolH[order - 1]);

  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&xp));
  PetscCallAbort(PETSC_COMM_WORLD, destroyReceiverFields(f));
  PetscCallAbort(PETSC_COMM_WORLD, destroyReceiverInterpolationMatrices(&Q));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&receivers));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemBackground(&bg));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&sigma));
  PetscCallAbort(PETSC_COMM_WORLD, destroyGrid(&grid));
  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&dm));
}

/**
 * @brief Checks receivers on an interface against the analytic field in the
 *        layer of their cell.
 *
 * Both the interpolated E_p,h and addCsemPrimaryAtReceivers() are checked.
 *
 * @param[in] order  Basis order.
 */
static void check_interface_receivers(PetscInt order) {
  const PetscInt                faces[3] = {2, 2, 2};
  const PetscReal               lower[3] = {880.0, -20.0, -520.0}, upper[3] = {920.0, 20.0, -480.0};
  const PetscReal               points[6] = {903.0, 7.0, -500.0, 889.0, -13.0, -500.0};
  const PetscReal               tolE[4]   = {1e-1, 2e-3, 1e-4, 2e-6};
  const PetscReal               interface = -500.0, sh[2] = {0.02, 1.0}, sv[2] = {0.01, 0.5};
  CsemSource                    src = {{0.0, 0.0, -450.0}, 1.0, 1.0, 35.0, 20.0};
  CsemSourceSet                 set = {freq, 1, &src};
  DM                            dm;
  Grid                          grid;
  Vec                           sigma, receivers, xp, fi[NUM_RECEIVER_FIELDS], fa[NUM_RECEIVER_FIELDS];
  CsemBackground                bg;
  CsemPrimary                   primary;
  ReceiverInterpolationMatrices Q;
  LayeredDipole                 dipole;
  PetscBool                    *all;
  PetscReal                     moment[3], errE = 0.0, errA = 0.0, jump = PETSC_MAX_REAL;
  PetscInt                      lo, hi;

  create_box_mesh(faces, lower, upper, 0.0, &dm);
  PetscCallAbort(PETSC_COMM_WORLD, setupNedelecGrid(order, PETGEM_BC_NATURAL, &dm, &grid));
  create_cell_conductivity(dm, unitSigma, &sigma);
  PetscCallAbort(PETSC_COMM_WORLD, PetscMemzero(&bg, sizeof(bg)));
  bg.type = CSEM_PRIMARY_LAYERED;
  PetscCallAbort(PETSC_COMM_WORLD, createLayeredModel(2, &interface, sh, sv, &bg.layered));
  PetscCallAbort(PETSC_COMM_WORLD, createCsemPrimary(&bg, dm, &grid, sigma, NULL, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, PetscMalloc1(grid.cellEnd - grid.cellStart, &all));
  for (PetscInt c = 0; c < grid.cellEnd - grid.cellStart; c++) all[c] = PETSC_TRUE;
  PetscCallAbort(PETSC_COMM_WORLD, computeCsemPrimary(1, &set, dm, &grid, all, &primary));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(all));

  receiver_vec(2, points, &receivers);
  PetscCallAbort(PETSC_COMM_WORLD, buildReceiverInterpolationMatrices(receivers, dm, &grid, &Q));
  PetscCallAbort(PETSC_COMM_WORLD, createReceiverFields(&Q, fi));
  PetscCallAbort(PETSC_COMM_WORLD, createReceiverFields(&Q, fa));
  primary_global(dm, &primary, 0, &xp);
  PetscCallAbort(PETSC_COMM_WORLD, interpolateReceiverFields(&Q, xp, iOmegaMu(freq), fi));
  for (PetscInt i = 0; i < NUM_RECEIVER_FIELDS; i++) PetscCallAbort(PETSC_COMM_WORLD, VecZeroEntries(fa[i]));
  PetscCallAbort(PETSC_COMM_WORLD, addCsemPrimaryAtReceivers(&primary, 0, 0, dm, &Q, receivers, fa));

  PetscCallAbort(PETSC_COMM_WORLD, computeVectorRotation(src.azimuthAngle, src.dipAngle, moment));
  PetscCallAbort(PETSC_COMM_WORLD, createLayeredDipole(&bg.layered, freq, src.position, moment, &dipole));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetOwnershipRange(fi[0], &lo, &hi));
  for (PetscInt r = lo; r < hi; r++) {
    const PetscInt     cellLayer = findLayer(&bg.layered, Q.cellCentroids[3 * r + 2]), other = 1 - cellLayer;
    const PetscInt     side = findLayerOnSide(&bg.layered, points[3 * r + 2], Q.cellCentroids[3 * r + 2]);
    PetscScalar        E[3], H[3], Eo[3];
    PetscReal          nE = 0.0, nH = 0.0, dE = 0.0, dA = 0.0;
    const PetscScalar *a;

    PT_CHECK(side == cellLayer, "receiver %d on the interface: layer %d, its cell lies in layer %d", (int)r, (int)side, (int)cellLayer);
    PetscCallAbort(PETSC_COMM_WORLD, evaluateLayeredDipole(&dipole, 1, &points[3 * r], &cellLayer, E, H));
    PetscCallAbort(PETSC_COMM_WORLD, evaluateLayeredDipole(&dipole, 1, &points[3 * r], &other, Eo, NULL));
    for (PetscInt d = 0; d < 3; d++) {
      nE = PetscMax(nE, PetscAbsScalar(E[d]));
      nH = PetscMax(nH, PetscAbsScalar(H[d]));
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(fi[RECV_EX + d], &a));
      dE = PetscMax(dE, PetscAbsScalar(a[r - lo] - E[d]));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(fi[RECV_EX + d], &a));
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(fa[RECV_EX + d], &a));
      dA = PetscMax(dA, PetscAbsScalar(a[r - lo] - E[d]));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(fa[RECV_EX + d], &a));
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(fa[RECV_HX + d], &a));
      dA = PetscMax(dA, PetscAbsScalar(a[r - lo] - H[d]) * nE / nH);
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(fa[RECV_HX + d], &a));
    }
    errE = PetscMax(errE, dE / nE);
    errA = PetscMax(errA, dA / nE);
    jump = PetscMin(jump, PetscAbsScalar(Eo[2] - E[2]) / nE);
  }
  PetscCallAbort(PETSC_COMM_WORLD, destroyLayeredDipole(&dipole));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &errE, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &errA, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &jump, 1, MPIU_REAL, MPIU_MIN, PETSC_COMM_WORLD));
  PetscCallAbort(PETSC_COMM_WORLD, PetscPrintf(PETSC_COMM_WORLD, "  order %d: interface receivers, E error %.3e, E_z jump %.3e\n",
                                               (int)order, (double)errE, (double)jump));
  PT_CHECK(jump > 5.0 * tolE[order - 1], "order %d: E_z jump %.3e across the interface below the check tolerance", (int)order,
           (double)jump);
  PT_CHECK(errE <= tolE[order - 1], "order %d: interface receivers, interpolated E relative error %.3e > %.1e", (int)order, (double)errE,
           (double)tolE[order - 1]);
  PT_CHECK(errA <= 1e-14, "order %d: interface receivers, addCsemPrimaryAtReceivers differs from the cell layer by %.3e", (int)order,
           (double)errA);

  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&xp));
  PetscCallAbort(PETSC_COMM_WORLD, destroyReceiverFields(fa));
  PetscCallAbort(PETSC_COMM_WORLD, destroyReceiverFields(fi));
  PetscCallAbort(PETSC_COMM_WORLD, destroyReceiverInterpolationMatrices(&Q));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&receivers));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemPrimary(&primary));
  PetscCallAbort(PETSC_COMM_WORLD, destroyCsemBackground(&bg));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&sigma));
  PetscCallAbort(PETSC_COMM_WORLD, destroyGrid(&grid));
  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&dm));
}

/**
 * @brief Runs the primary field checks and reports the number of failures.
 *
 * @param[in] argc  Argument count.
 * @param[in] argv  Argument vector; argv[1] is the basis order where the
 *                  harness takes one.
 *
 * @return int 0 when every check passes, non-zero otherwise.
 */
int main(int argc, char **argv) {
  PetscInt order = 1;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  if (argc > 1) order = atoi(argv[1]);
  PetscCheck(order >= 1 && order <= 4, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "test_csem_primary: order %d outside 1-4", (int)order);
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_type", "preonly"));
  PetscCall(PetscOptionsSetValue(NULL, "-pc_type", "lu"));
  PetscCall(PetscOptionsSetValue(NULL, "-pc_factor_mat_solver_type", "mumps"));

  check_background(order);
  check_rhs(order);
  check_model_decomposition(order);
  check_layered_receivers(order);
  check_layered_conventions(order);
  check_interface_receivers(order);

  int rc = pt_report("test_csem_primary");
  PetscCall(PetscFinalize());
  return rc;
}
