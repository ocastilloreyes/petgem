/*
 * Filename: test_mt_rhs.c
 * Author: PETGEM test suite
 * Date: 2026-10-09
 *
 * Description:
 * MT boundary right-hand side (assembleMtBoundaryRHS). For a given order
 * (argv[1]) and a field F in the Nedelec space, with coefficients c from the
 * L2 projection Ms c = INT F . N (Ms from assembleMaxwellOperator, sigma = 1),
 * checks
 *     c^T b = -iωμ ∮ F . (n x Ĥ) dΓ
 * against the closed form on a box with H(z) linear (0 at z_min, 1 at z_max):
 *   order 1:  F = (1,0,0), (0,0,1) for x-polarization; (0,1,0) for y-polarization;
 *   order 2+: F = (z,0,x) for x-polarization; (0,z,y) for y-polarization.
 * Also checks the size of B and that a PEC grid is rejected.
 *
 * Usage: [mpirun -n N] test_mt_rhs <order>   (order in 1..6).  Exit 0 iff all checks pass.
 */
#include "assembly.h"
#include "fem.h"
#include "grid.h"
#include "mt.h"
#include "constants.h"
#include "petgem_test.h"
#include "box_mesh.h"
#include <petsc.h>

static const PetscInt  boxFaces[3] = {3, 2, 3};
static const PetscReal boxLower[3] = {0.0, 0.0, -1.0};
static const PetscReal boxUpper[3] = {3.0, 2.0, 0.5};

typedef void (*FieldFn)(const PetscReal x[3], PetscReal F[3]);

static void f_x(const PetscReal x[3], PetscReal F[3])   { (void)x; F[0] = 1.0; F[1] = 0.0; F[2] = 0.0; }
static void f_y(const PetscReal x[3], PetscReal F[3])   { (void)x; F[0] = 0.0; F[1] = 1.0; F[2] = 0.0; }
static void f_z(const PetscReal x[3], PetscReal F[3])   { (void)x; F[0] = 0.0; F[1] = 0.0; F[2] = 1.0; }
static void f_zox(const PetscReal x[3], PetscReal F[3]) { F[0] = x[2]; F[1] = 0.0; F[2] = x[0]; }
static void f_ozy(const PetscReal x[3], PetscReal F[3]) { F[0] = 0.0; F[1] = x[2]; F[2] = x[1]; }

static PetscReal unit_sigma(const PetscReal c[3]) { (void)c; return 1.0; }

/* Coefficients c of F: solve Ms c = INT F . N */
static void project(const petgemParams params, DM dm, Grid grid, Mat Ms, FieldFn F, Vec *c)
{
  Vec                    f;
  ISLocalToGlobalMapping mapping;
  PetscSection           section;
  Quadrature3D           quad;
  PetscReal            **Ni;
  PetscScalar           *closure;
  PetscInt               numIdx, *idx;
  KSP                    ksp;

  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(dm, &f));
  PetscCallAbort(PETSC_COMM_WORLD, DMGetLocalToGlobalMapping(dm, &mapping));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetLocalToGlobalMapping(f, mapping));
  PetscCallAbort(PETSC_COMM_WORLD, VecZeroEntries(f));
  PetscCallAbort(PETSC_COMM_WORLD, DMGetLocalSection(dm, &section));

  PetscCallAbort(PETSC_COMM_WORLD, computeNum3DQuadraturePoints(params.order, &quad));
  PetscCallAbort(PETSC_COMM_WORLD, PetscCalloc1(quad.numPoints, &quad.points));
  for (PetscInt i = 0; i < quad.numPoints; i++) PetscCallAbort(PETSC_COMM_WORLD, PetscCalloc1(3, &quad.points[i]));
  PetscCallAbort(PETSC_COMM_WORLD, PetscCalloc1(quad.numPoints, &quad.weights));
  PetscCallAbort(PETSC_COMM_WORLD, compute3DQuadraturePoints(&quad));
  PetscCallAbort(PETSC_COMM_WORLD, PetscCalloc1(grid.numDofInCell, &closure));
  PetscCallAbort(PETSC_COMM_WORLD, PetscCalloc1(3, &Ni));
  for (PetscInt d = 0; d < 3; d++) PetscCallAbort(PETSC_COMM_WORLD, PetscCalloc1(grid.numDofInCell, &Ni[d]));

  for (PetscInt cid = grid.cellStart; cid < grid.cellEnd; cid++) {
    Cell cell;
    PetscCallAbort(PETSC_COMM_WORLD, extractCellCoordinates(dm, cid, &cell));
    PetscCallAbort(PETSC_COMM_WORLD, computeCellJacobian(&cell));
    for (PetscInt j = 0; j < grid.numDofInCell; j++) closure[j] = 0.0;
    for (PetscInt q = 0; q < quad.numPoints; q++) {
      const PetscReal *ref = quad.points[q];
      PetscReal        x[3], Fq[3];
      for (PetscInt d = 0; d < 3; d++)
        x[d] = cell.coordinates[d] + cell.jacobian[0][d] * ref[0] + cell.jacobian[1][d] * ref[1] + cell.jacobian[2][d] * ref[2];
      F(x, Fq);
      PetscCallAbort(PETSC_COMM_WORLD, evaluateNedelecBasis(&grid.fem, &cell, ref, Ni, NULL));
      for (PetscInt j = 0; j < grid.numDofInCell; j++)
        closure[j] += quad.weights[q] * cell.detJacobian * (Ni[0][j] * Fq[0] + Ni[1][j] * Fq[1] + Ni[2][j] * Fq[2]);
    }
    PetscCallAbort(PETSC_COMM_WORLD, DMPlexGetClosureIndices(dm, section, section, cid, PETSC_TRUE, &numIdx, &idx, NULL, NULL));
    PetscCallAbort(PETSC_COMM_WORLD, VecSetValuesLocal(f, numIdx, idx, closure, ADD_VALUES));
    PetscCallAbort(PETSC_COMM_WORLD, DMPlexRestoreClosureIndices(dm, section, section, cid, PETSC_TRUE, &numIdx, &idx, NULL, NULL));
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(f));
  PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(f));

  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f, c));
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp, Ms, Ms));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp, KSPPREONLY));
  PC pc;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp, &pc));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCLU));
  PetscCallAbort(PETSC_COMM_WORLD, PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(ksp, f, *c));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&ksp));

  for (PetscInt d = 0; d < 3; d++) PetscCallAbort(PETSC_COMM_WORLD, PetscFree(Ni[d]));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(Ni));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(closure));
  for (PetscInt i = 0; i < quad.numPoints; i++) PetscCallAbort(PETSC_COMM_WORLD, PetscFree(quad.points[i]));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(quad.points));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(quad.weights));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&f));
}

static void check_identity(const petgemParams params, DM dm, Grid grid, Mat Ms, Mat B, PetscInt pol, FieldFn F,
                           PetscScalar expected, const char *tag)
{
  Vec         c, b;
  PetscScalar value;

  project(params, dm, grid, Ms, F, &c);
  PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetColumnVecRead(B, pol, &b));
  PetscCallAbort(PETSC_COMM_WORLD, VecTDot(b, c, &value));
  PetscCallAbort(PETSC_COMM_WORLD, MatDenseRestoreColumnVecRead(B, pol, &b));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&c));

  const PetscReal scale = PetscMax(PetscAbsScalar(expected), 1.0e-6);
  PT_CLOSE(PetscAbsScalar(value - expected) / scale, 0.0, 1e-9, "[%s] order=%d: c^T b = %.12e%+.12ei, expected %.12e%+.12ei", tag,
           (int)params.order, (double)PetscRealPart(value), (double)PetscImaginaryPart(value), (double)PetscRealPart(expected),
           (double)PetscImaginaryPart(expected));
}

int main(int argc, char **argv)
{
  PetscMPIInt  rank;
  petgemParams params;
  DM           dm;
  Grid         grid;
  IS           faces;
  MtBoxFace   *tags;
  Vec          sigma;
  Mat          B, K, Ms;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscMemzero(&params, sizeof(params)));
  params.order = (argc > 1) ? atoi(argv[1]) : 1;

  create_box_mesh(boxFaces, boxLower, boxUpper, 0.0, &dm);
  PetscCall(setupNedelecGrid(params, PETGEM_BC_NATURAL, &dm, &grid));
  PetscCall(getBoundaryFaces(dm, &faces));
  PetscCall(classifyMtBoxFaces(dm, faces, &tags));
  create_cell_conductivity(dm, unit_sigma, &sigma);

  /* H(z) linear: 0 at z_min, 1 at z_max */
  PetscReal   zNodes[2] = {boxLower[2], boxUpper[2]};
  PetscScalar hNodes[2] = {0.0, 1.0};
  Mt1DField   field     = {2, zNodes, hNodes};

  const PetscReal omega = 2.0 * PETSC_PI * 3.0;
  PetscCall(assembleMtBoundaryRHS(params, omega, dm, grid, faces, tags, &field, &B));
  PetscCall(assembleMaxwellOperator(params, dm, grid, sigma, 0.0, &K, &Ms, NULL));

  /* Positively oriented cells */
  for (PetscInt cid = grid.cellStart; cid < grid.cellEnd; cid++) {
    Cell cell;
    PetscCall(extractCellCoordinates(dm, cid, &cell));
    PetscCall(computeCellJacobian(&cell));
    PT_CHECK(cell.detJacobian > 0.0, "cell %d: detJ = %g", (int)cid, (double)cell.detJacobian);
  }

  /* Size of B */
  {
    Vec      v;
    PetscInt n, rows, cols;
    PetscCall(DMCreateGlobalVector(dm, &v));
    PetscCall(VecGetSize(v, &n));
    PetscCall(VecDestroy(&v));
    PetscCall(MatGetSize(B, &rows, &cols));
    PT_CHECK(rows == n && cols == MT_NUM_POLARIZATIONS, "B is %d x %d, expected %d x %d", (int)rows, (int)cols, (int)n, MT_NUM_POLARIZATIONS);
  }

  const PetscReal   lx = boxUpper[0] - boxLower[0], ly = boxUpper[1] - boxLower[1], lz = boxUpper[2] - boxLower[2];
  const PetscReal   aTop = lx * ly, zTop = boxUpper[2], intH = 0.5 * lz;
  const PetscScalar iwm  = PETSC_i * omega * MU;

  if (params.order == 1) {
    check_identity(params, dm, grid, Ms, B, 0, f_x, iwm * aTop, "x-pol, F = (1,0,0)");
    check_identity(params, dm, grid, Ms, B, 0, f_z, 0.0, "x-pol, F = (0,0,1)");
    check_identity(params, dm, grid, Ms, B, 1, f_y, -iwm * aTop, "y-pol, F = (0,1,0)");
  } else {
    check_identity(params, dm, grid, Ms, B, 0, f_zox, -iwm * (-zTop * aTop + lx * ly * intH), "x-pol, F = (z,0,x)");
    check_identity(params, dm, grid, Ms, B, 1, f_ozy, -iwm * (zTop * aTop - lx * ly * intH), "y-pol, F = (0,z,y)");
  }

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&K));
  PetscCall(MatDestroy(&Ms));

  /* PEC grid: rejected */
  {
    DM             dmPec;
    Grid           gridPec;
    IS             facesPec;
    MtBoxFace     *tagsPec;
    Mat            Bpec = NULL;
    PetscErrorCode ierr;
    create_box_mesh(boxFaces, boxLower, boxUpper, 0.0, &dmPec);
    PetscCall(setupNedelecGrid(params, PETGEM_BC_PEC, &dmPec, &gridPec));
    PetscCall(getBoundaryFaces(dmPec, &facesPec));
    PetscCall(classifyMtBoxFaces(dmPec, facesPec, &tagsPec));
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    ierr = assembleMtBoundaryRHS(params, omega, dmPec, gridPec, facesPec, tagsPec, &field, &Bpec);
    PetscCall(PetscPopErrorHandler());
    PT_CHECK(ierr != PETSC_SUCCESS, "PEC grid not rejected");
    PetscCall(MatDestroy(&Bpec));
    PetscCall(PetscFree(tagsPec));
    PetscCall(ISDestroy(&facesPec));
    PetscCall(DMDestroy(&gridPec.H1dm));
    PetscCall(DMDestroy(&dmPec));
  }

  PetscCall(VecDestroy(&sigma));
  PetscCall(PetscFree(tags));
  PetscCall(ISDestroy(&faces));
  PetscCall(DMDestroy(&grid.H1dm));
  PetscCall(DMDestroy(&dm));

  int localFail = pt_failures, anyFail = 0;
  PetscCallMPI(MPI_Allreduce(&localFail, &anyFail, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
  int rc = 0;
  if (rank == 0) rc = pt_report("test_mt_rhs");
  else if (pt_failures) fprintf(stderr, "rank %d: %d failures\n", rank, pt_failures);
  PetscCall(PetscFinalize());
  return (rc || anyFail) ? 1 : 0;
}
