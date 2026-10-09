/*
 * Filename: test_mt1d.c
 * Author: PETGEM test suite
 * Date: 2026-10-09
 *
 * Description:
 * MT 1D boundary field. Checks:
 *   - solveMt1D against the exact layered solution (transfer of H and H',
 *     or H and ρH' for the 'h' equation), for both 1D equations, with
 *     second-order convergence in the 1D element size;
 *   - evalMt1DField interpolation and clamping;
 *   - classifyMtBoxFaces on a box: every face tagged on its plane; a moved
 *     top vertex is rejected;
 *   - buildMt1DProfile on a 3-layer box: layers, sigma and edge length, the
 *     same on every rank; a lateral anomaly is rejected.
 *
 * Usage: [mpirun -n N] test_mt1d   Exit 0 iff all checks pass.
 */
#include "grid.h"
#include "mt.h"
#include "constants.h"
#include "petgem_test.h"
#include "box_mesh.h"
#include <petsc.h>

/* 1D model: [-3000, -1000] σ = 1, [-1000, 0] σ = 0.01, [0, 1000] air */
static const PetscReal modelZ[4]     = {-3000.0, -1000.0, 0.0, 1000.0};
static const PetscReal modelSigma[3] = {1.0, 0.01, 1.0e-8};
static const PetscReal modelH[3]     = {100.0, 100.0, 100.0};

/* Box mesh: 3 x 2 x 3 hexahedra on [0,3] x [0,2] x [-1,0.5]; layers by cell centroid z */
static const PetscInt  boxFaces[3]    = {3, 2, 3};
static const PetscReal boxLower[3]    = {0.0, 0.0, -1.0};
static const PetscReal boxUpper[3]    = {3.0, 2.0, 0.5};
static const PetscReal boxLayerZ[4]   = {-1.0, -0.5, 0.0, 0.5};
static const PetscReal boxSigma[3]    = {0.1, 1.0, 1.0e-8};

/* Exact H(z): propagate (H, H') upward from H(z_min) = 0, H'(z_min) = 1, then scale to H(z_max) = 1. */
static PetscScalar exact_h(const Mt1DProfile *p, Mt1DEquation eq, PetscReal omega, PetscReal z)
{
  PetscScalar H = 0.0, dH = 1.0, Htop = 0.0;
  PetscScalar Hz = 0.0;
  for (PetscInt l = 0; l < p->numLayers; l++) {
    const PetscScalar k = PetscSqrtScalar(PETSC_i * omega * MU * p->sigma[l]);
    if (l > 0 && eq == MT_1D_EQUATION_H) dH *= p->sigma[l] / p->sigma[l - 1];
    if (z >= p->z[l] && z <= p->z[l + 1]) {
      const PetscReal d = z - p->z[l];
      Hz = H * PetscCosComplex(k * d) + dH / k * PetscSinComplex(k * d);
    }
    const PetscReal   d  = p->z[l + 1] - p->z[l];
    const PetscScalar H1 = H * PetscCosComplex(k * d) + dH / k * PetscSinComplex(k * d);
    const PetscScalar d1 = -H * k * PetscSinComplex(k * d) + dH * PetscCosComplex(k * d);
    H = H1;
    dH = d1;
  }
  Htop = H;
  return Hz / Htop;
}

static PetscReal field_error(const Mt1DField *f, const Mt1DProfile *p, Mt1DEquation eq, PetscReal omega)
{
  PetscReal err = 0.0;
  for (PetscInt i = 0; i < f->numNodes; i++) err = PetscMax(err, PetscAbsScalar(f->H[i] - exact_h(p, eq, omega, f->z[i])));
  return err;
}

static void check_solve(Mt1DEquation eq, const char *tag)
{
  Mt1DProfile p;
  Mt1DField   f10, f20;
  MtParams    mt;
  const PetscReal omega = 2.0 * PETSC_PI * 1.0;

  p.numLayers = 3;
  p.z = (PetscReal *)modelZ;
  p.sigma = (PetscReal *)modelSigma;
  p.h = (PetscReal *)modelH;
  mt.equation1D = eq;

  mt.refine1D = 10;
  PetscCallAbort(PETSC_COMM_SELF, solveMt1D(&p, &mt, omega, &f10));
  mt.refine1D = 20;
  PetscCallAbort(PETSC_COMM_SELF, solveMt1D(&p, &mt, omega, &f20));

  PT_CHECK(f10.numNodes == 401, "[%s] %d nodes at refine 10, expected 401", tag, (int)f10.numNodes);
  PT_CLOSE(f10.z[0], modelZ[0], 1e-12, "[%s] first node %g", tag, f10.z[0]);
  PT_CLOSE(f10.z[f10.numNodes - 1], modelZ[3], 1e-12, "[%s] last node %g", tag, f10.z[f10.numNodes - 1]);
  PT_CLOSE(PetscAbsScalar(f10.H[0]), 0.0, 1e-14, "[%s] H(z_min) = %g", tag, PetscAbsScalar(f10.H[0]));
  PT_CLOSE(PetscAbsScalar(f10.H[f10.numNodes - 1] - 1.0), 0.0, 1e-14, "[%s] H(z_max) != 1", tag);
  for (PetscInt l = 1; l < 3; l++) {
    PetscBool found = PETSC_FALSE;
    for (PetscInt i = 0; i < f10.numNodes; i++) if (f10.z[i] == modelZ[l]) found = PETSC_TRUE;
    PT_CHECK(found, "[%s] no node at interface z = %g", tag, modelZ[l]);
  }

  const PetscReal e10 = field_error(&f10, &p, eq, omega);
  const PetscReal e20 = field_error(&f20, &p, eq, omega);
  PT_CHECK(e10 < 1e-3, "[%s] max |H - H_exact| = %.3e at refine 10", tag, e10);
  PT_CHECK(e10 / e20 > 3.5 && e10 / e20 < 4.5, "[%s] convergence ratio %.3f (expected ~4)", tag, e10 / e20);
  PetscCallAbort(PETSC_COMM_WORLD, PetscPrintf(PETSC_COMM_WORLD, "[%s] max error refine 10: %.3e, refine 20: %.3e, ratio %.2f\n", tag, (double)e10, (double)e20, (double)(e10 / e20)));

  /* Interpolation and clamping */
  PetscScalar H;
  PetscCallAbort(PETSC_COMM_SELF, evalMt1DField(&f10, f10.z[7], &H));
  PT_CLOSE(PetscAbsScalar(H - f10.H[7]), 0.0, 1e-14, "[%s] eval at a node", tag);
  PetscCallAbort(PETSC_COMM_SELF, evalMt1DField(&f10, 0.5 * (f10.z[7] + f10.z[8]), &H));
  PT_CLOSE(PetscAbsScalar(H - 0.5 * (f10.H[7] + f10.H[8])), 0.0, 1e-14, "[%s] eval at a midpoint", tag);
  PetscCallAbort(PETSC_COMM_SELF, evalMt1DField(&f10, modelZ[3] + 10.0, &H));
  PT_CLOSE(PetscAbsScalar(H - 1.0), 0.0, 1e-14, "[%s] eval above z_max", tag);
  PetscCallAbort(PETSC_COMM_SELF, evalMt1DField(&f10, modelZ[0] - 10.0, &H));
  PT_CLOSE(PetscAbsScalar(H), 0.0, 1e-14, "[%s] eval below z_min", tag);

  PetscCallAbort(PETSC_COMM_SELF, destroyMt1DField(&f10));
  PetscCallAbort(PETSC_COMM_SELF, destroyMt1DField(&f20));
}

static void check_equations_differ(void)
{
  Mt1DProfile p;
  Mt1DField   fp, fh;
  MtParams    mt = {10, MT_1D_EQUATION_PAPER};
  const PetscReal omega = 2.0 * PETSC_PI * 1.0;

  p.numLayers = 3;
  p.z = (PetscReal *)modelZ;
  p.sigma = (PetscReal *)modelSigma;
  p.h = (PetscReal *)modelH;
  PetscCallAbort(PETSC_COMM_SELF, solveMt1D(&p, &mt, omega, &fp));
  mt.equation1D = MT_1D_EQUATION_H;
  PetscCallAbort(PETSC_COMM_SELF, solveMt1D(&p, &mt, omega, &fh));
  PetscReal diff = 0.0;
  for (PetscInt i = 0; i < fp.numNodes; i++) diff = PetscMax(diff, PetscAbsScalar(fp.H[i] - fh.H[i]));
  PT_CHECK(diff > 1e-2, "paper and h equations coincide on a layered model (max diff %.3e)", diff);
  PetscCallAbort(PETSC_COMM_SELF, destroyMt1DField(&fp));
  PetscCallAbort(PETSC_COMM_SELF, destroyMt1DField(&fh));
}

static PetscReal layered_sigma(const PetscReal c[3])
{
  PetscInt layer = 0;
  while (layer < 2 && c[2] > boxLayerZ[layer + 1]) layer++;
  return boxSigma[layer];
}

static PetscReal anomaly_sigma(const PetscReal c[3])
{
  const PetscReal s = layered_sigma(c);
  return (s == boxSigma[1] && c[0] < 1.0 && c[1] < 1.0) ? 5.0 : s;
}

static void setup_box(PetscReal topShift, DM *dm, Grid *grid, IS *faces)
{
  petgemParams params;
  PetscCallAbort(PETSC_COMM_WORLD, PetscMemzero(&params, sizeof(params)));
  params.order = 1;
  create_box_mesh(boxFaces, boxLower, boxUpper, topShift, dm);
  PetscCallAbort(PETSC_COMM_WORLD, setupNedelecGrid(params, PETGEM_BC_NATURAL, dm, grid));
  PetscCallAbort(PETSC_COMM_WORLD, getBoundaryFaces(*dm, faces));
}

static void check_box(void)
{
  DM              dm;
  Grid            grid;
  IS              faces;
  MtBoxFace      *tags;
  const PetscInt *fidx;
  PetscInt        nf, localCount[7] = {0}, count[7];

  setup_box(0.0, &dm, &grid, &faces);
  PetscCallAbort(PETSC_COMM_WORLD, classifyMtBoxFaces(dm, faces, &tags));
  PetscCallAbort(PETSC_COMM_WORLD, ISGetLocalSize(faces, &nf));
  PetscCallAbort(PETSC_COMM_WORLD, ISGetIndices(faces, &fidx));
  for (PetscInt i = 0; i < nf; i++) {
    PetscInt  cell;
    PetscReal vtx[NUM_VERTICES_PER_FACE][NUM_DIMENSIONS], n[NUM_DIMENSIONS], area;
    PetscCallAbort(PETSC_COMM_WORLD, computeBoundaryFaceGeometry(dm, fidx[i], &cell, vtx, n, &area));
    const PetscReal c[3] = {(vtx[0][0] + vtx[1][0] + vtx[2][0]) / 3, (vtx[0][1] + vtx[1][1] + vtx[2][1]) / 3, (vtx[0][2] + vtx[1][2] + vtx[2][2]) / 3};
    PetscBool ok = PETSC_FALSE;
    switch (tags[i]) {
      case MT_FACE_TOP:    ok = (PetscBool)(PetscAbsReal(c[2] - boxUpper[2]) < 1e-12); break;
      case MT_FACE_BOTTOM: ok = (PetscBool)(PetscAbsReal(c[2] - boxLower[2]) < 1e-12); break;
      case MT_FACE_XMIN:   ok = (PetscBool)(PetscAbsReal(c[0] - boxLower[0]) < 1e-12); break;
      case MT_FACE_XMAX:   ok = (PetscBool)(PetscAbsReal(c[0] - boxUpper[0]) < 1e-12); break;
      case MT_FACE_YMIN:   ok = (PetscBool)(PetscAbsReal(c[1] - boxLower[1]) < 1e-12); break;
      case MT_FACE_YMAX:   ok = (PetscBool)(PetscAbsReal(c[1] - boxUpper[1]) < 1e-12); break;
    }
    PT_CHECK(ok, "face %d tagged Gamma_%d off its plane", (int)fidx[i], (int)tags[i]);
    localCount[tags[i]]++;
  }
  PetscCallAbort(PETSC_COMM_WORLD, ISRestoreIndices(faces, &fidx));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(localCount, count, 7, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  const PetscInt nx = boxFaces[0], ny = boxFaces[1], nz = boxFaces[2];
  const PetscInt expected[7] = {0, 2 * nx * ny, 2 * nx * nz, 2 * ny * nz, 2 * nx * nz, 2 * ny * nz, 2 * nx * ny};
  for (PetscInt g = 1; g <= 6; g++) PT_CHECK(count[g] == expected[g], "Gamma_%d has %d faces, expected %d", (int)g, (int)count[g], (int)expected[g]);

  /* Profile of the layered box */
  Vec         sigma;
  Mt1DProfile p;
  create_cell_conductivity(dm, layered_sigma, &sigma);
  PetscCallAbort(PETSC_COMM_WORLD, buildMt1DProfile(dm, sigma, faces, tags, &p));
  PT_CHECK(p.numLayers == 3, "%d layers, expected 3", (int)p.numLayers);
  if (p.numLayers == 3) {
    for (PetscInt l = 0; l <= 3; l++) PT_CLOSE(p.z[l], boxLayerZ[l], 1e-12, "interface %d at %g, expected %g", (int)l, p.z[l], boxLayerZ[l]);
    for (PetscInt l = 0; l < 3; l++) {
      PT_CLOSE(p.sigma[l], boxSigma[l], 1e-15, "layer %d sigma %g, expected %g", (int)l, p.sigma[l], boxSigma[l]);
      PT_CLOSE(p.h[l], 0.5, 1e-12, "layer %d edge length %g, expected 0.5", (int)l, p.h[l]);
    }
  }
  PetscCallAbort(PETSC_COMM_WORLD, destroyMt1DProfile(&p));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&sigma));

  /* Lateral anomaly: rejected */
  PetscErrorCode ierr;
  create_cell_conductivity(dm, anomaly_sigma, &sigma);
  PetscCallAbort(PETSC_COMM_WORLD, PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = buildMt1DProfile(dm, sigma, faces, tags, &p);
  PetscCallAbort(PETSC_COMM_WORLD, PetscPopErrorHandler());
  PT_CHECK(ierr != PETSC_SUCCESS, "lateral anomaly not rejected");
  if (ierr == PETSC_SUCCESS) PetscCallAbort(PETSC_COMM_WORLD, destroyMt1DProfile(&p));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&sigma));

  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(tags));
  PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&faces));
  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&grid.H1dm));
  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&dm));
}

static void check_not_a_box(void)
{
  DM         dm;
  Grid       grid;
  IS         faces;
  MtBoxFace *tags = NULL;
  int        failedLocal, failedAny;

  setup_box(-0.05, &dm, &grid, &faces);
  PetscCallAbort(PETSC_COMM_WORLD, PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  failedLocal = (classifyMtBoxFaces(dm, faces, &tags) != PETSC_SUCCESS);
  PetscCallAbort(PETSC_COMM_WORLD, PetscPopErrorHandler());
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(&failedLocal, &failedAny, 1, MPI_INT, MPI_LOR, PETSC_COMM_WORLD));
  PT_CHECK(failedAny, "moved top vertex not rejected");
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(tags));
  PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&faces));
  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&grid.H1dm));
  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&dm));
}

int main(int argc, char **argv)
{
  PetscMPIInt rank;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  check_solve(MT_1D_EQUATION_PAPER, "paper");
  check_solve(MT_1D_EQUATION_H, "h");
  check_equations_differ();
  check_box();
  check_not_a_box();

  int localFail = pt_failures, anyFail = 0;
  PetscCallMPI(MPI_Allreduce(&localFail, &anyFail, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
  int rc = 0;
  if (rank == 0) rc = pt_report("test_mt1d");
  else if (pt_failures) fprintf(stderr, "rank %d: %d failures\n", rank, pt_failures);
  PetscCall(PetscFinalize());
  return (rc || anyFail) ? 1 : 0;
}
