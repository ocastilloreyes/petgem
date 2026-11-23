/*
 * Filename: test_elements.c
 * Author: PETGEM test suite
 * Date: 2026-07-03
 *
 * Description:
 * LEVEL 3 - Element-level matrix tests. Builds the elemental mass (Me) and
 * stiffness (Ke) matrices and the per-cell discrete gradient (G) through the
 * unchanged production entry points (computeElementalMatrices,
 * buildDiscreteGradientMatrix) for a given order (argv[1]) on two cells (the
 * reference tetrahedron and a skewed one), then checks:
 *   - dimensions and finiteness;
 *   - symmetry of Me and Ke;
 *   - positive-definiteness of Me (x^T Me x > 0) and positive-semidefiniteness
 *     of Ke (x^T Ke x >= 0);
 *   - the De Rham identity Ke . G == 0 (the discrete gradient lies in the
 *     kernel of the curl-curl stiffness);
 *   - a positively oriented, non-degenerate cell Jacobian;
 *   - tetrahedronXyzToReference inverts x = v0 + J^T r, the frame of
 *     evaluateNedelecBasis;
 *   - interpolateNedelecCell of grad(phi_j) equals column j of G;
 *   - interpolateNedelecCell reproduces a complex field of degree order - 1
 *     plus b x x.
 *
 * Usage: test_elements <order>   (order in 1..6).  Exit 0 iff all checks pass.
 */
#include "fem.h"
#include "fe_nodal.h"
#include "grid.h"
#include "constants.h"
#include "petgem_test.h"
#include <petsc.h>

/* Cell and order seen by the test fields. */
typedef struct {
  PetscInt         order;
  const PetscReal *coords;
  PetscReal        invJacobian[3][3];
} FieldCtx;

static const PetscReal polyAlpha[3]   = {0.3, -0.2, 0.5};
static const PetscReal polyBeta[3][3] = {{0.7, -0.4, 0.2}, {0.1, 0.6, -0.3}, {-0.5, 0.2, 0.8}};
static const PetscReal polyB[3]       = {0.4, -0.7, 0.25};

/* Field j = grad(phi_j), phi_j the H1 basis function j of the cell. */
static PetscErrorCode gradientFields(PetscInt numPoints, const PetscReal points[], PetscInt numFields, PetscScalar values[], void *ctx) {
  PetscFunctionBeginUser;
  const FieldCtx *c = (const FieldCtx *)ctx;
  PetscReal       ShapH[FEM_MAX_H1_DOF], gradHbuf[3][FEM_MAX_H1_DOF], ref[3];
  PetscReal      *GradH[3] = {gradHbuf[0], gradHbuf[1], gradHbuf[2]};

  for (PetscInt q = 0; q < numPoints; q++) {
    PetscCall(tetrahedronXyzToReference(c->coords, &points[q * 3], ref));
    PetscCall(feNodalH1Shape(c->order, c->coords, ref, ShapH, GradH));
    for (PetscInt j = 0; j < numFields; j++)
      for (PetscInt d = 0; d < 3; d++)
        values[(j * numPoints + q) * 3 + d] =
            c->invJacobian[d][0] * GradH[0][j] + c->invJacobian[d][1] * GradH[1][j] + c->invJacobian[d][2] * GradH[2][j];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* F_d(x) = (1 + 0.5i) (alpha_d + beta_d . x)^(order-1) + (b x x)_d */
static void polyField(PetscInt order, const PetscReal x[3], PetscScalar F[3]) {
  const PetscReal bx[3] = {polyB[1] * x[2] - polyB[2] * x[1], polyB[2] * x[0] - polyB[0] * x[2], polyB[0] * x[1] - polyB[1] * x[0]};
  for (PetscInt d = 0; d < 3; d++) {
    const PetscReal s = polyAlpha[d] + polyBeta[d][0] * x[0] + polyBeta[d][1] * x[1] + polyBeta[d][2] * x[2];
    F[d]              = (1.0 + 0.5 * PETSC_i) * PetscPowRealInt(s, order - 1) + bx[d];
  }
}

/* Every field is polyField. */
static PetscErrorCode polyFields(PetscInt numPoints, const PetscReal points[], PetscInt numFields, PetscScalar values[], void *ctx) {
  PetscFunctionBeginUser;
  const FieldCtx *c = (const FieldCtx *)ctx;
  for (PetscInt f = 0; f < numFields; f++)
    for (PetscInt q = 0; q < numPoints; q++) polyField(c->order, &points[q * 3], &values[(f * numPoints + q) * 3]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* interpolateNedelecCell against G and against polyField on one cell. */
static void check_interpolation(PetscInt order, const PetscReal coords[12], PetscReal **G, const char *tag) {
  const PetscInt dof = order * (order + 2) * (order + 3) / 2;
  const PetscInt nH1 = (order + 1) * (order + 2) * (order + 3) / 6;

  FemSpace fem;
  PetscCallAbort(PETSC_COMM_SELF, PetscMemzero(&fem, sizeof(fem)));
  fem.order        = order;
  fem.numDofInCell = dof;

  Cell cell;
  PetscCallAbort(PETSC_COMM_SELF, PetscMemzero(&cell, sizeof(cell)));
  for (int i = 0; i < 12; i++) cell.coordinates[i] = coords[i];
  PetscCallAbort(PETSC_COMM_SELF, computeCellJacobian(&cell));

  FieldCtx ctx;
  ctx.order  = order;
  ctx.coords = coords;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) ctx.invJacobian[i][j] = cell.invJacobian[i][j];

  PetscScalar *coeff;
  PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(nH1 * dof, &coeff));

  /* interpolate(grad phi_j) == G[:, j] */
  PetscCallAbort(PETSC_COMM_SELF, interpolateNedelecCell(&fem, &cell, gradientFields, &ctx, nH1, coeff));
  PetscReal gErr = 0;
  for (PetscInt j = 0; j < nH1; j++)
    for (PetscInt i = 0; i < dof; i++) gErr = PetscMax(gErr, PetscAbsScalar(coeff[j * dof + i] - G[i][j]));
  PT_CHECK(gErr < 1e-10, "[%s] order=%d: max |interpolate(grad phi) - G| = %.3e", tag, (int)order, gErr);

  /* sum_i c_i N_i(x) == F(x) at interior points */
  PetscCallAbort(PETSC_COMM_SELF, interpolateNedelecCell(&fem, &cell, polyFields, &ctx, 1, coeff));
  PetscReal     **Ni;
  const PetscReal refPoints[4][3] = {{0.25, 0.25, 0.25}, {0.1, 0.2, 0.3}, {0.6, 0.1, 0.05}, {0.05, 0.7, 0.2}};
  PetscReal       fErr = 0, fMax = 0;
  PetscCallAbort(PETSC_COMM_SELF, createBasisValues(dof, &Ni));
  for (int p = 0; p < 4; p++) {
    PetscReal   x[3];
    PetscScalar F[3];
    for (int d = 0; d < 3; d++)
      x[d] = coords[d] + refPoints[p][0] * (coords[3 + d] - coords[d]) + refPoints[p][1] * (coords[6 + d] - coords[d]) +
             refPoints[p][2] * (coords[9 + d] - coords[d]);
    polyField(order, x, F);
    PetscCallAbort(PETSC_COMM_SELF, evaluateNedelecBasis(&fem, &cell, refPoints[p], Ni, NULL));
    for (int d = 0; d < 3; d++) {
      PetscScalar u = 0;
      for (PetscInt i = 0; i < dof; i++) u += coeff[i] * Ni[d][i];
      fErr = PetscMax(fErr, PetscAbsScalar(u - F[d]));
      fMax = PetscMax(fMax, PetscAbsScalar(F[d]));
    }
  }
  PT_CHECK(fErr < 1e-10 * (1.0 + fMax), "[%s] order=%d: max |interpolant - F| = %.3e (max |F| = %.3e)", tag, (int)order, fErr, fMax);

  PetscCallAbort(PETSC_COMM_SELF, destroyBasisValues(&Ni));
  PetscCallAbort(PETSC_COMM_SELF, PetscFree(coeff));
}

static PetscReal **alloc2d(PetscInt r, PetscInt c) {
  PetscReal **m, *flat;
  PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(r, &m));
  PetscCallAbort(PETSC_COMM_SELF, PetscCalloc1(r * c, &flat));
  for (PetscInt i = 0; i < r; i++) m[i] = flat + i * c;
  return m;
}

static void free2d(PetscReal **m) {
  PetscCallAbort(PETSC_COMM_SELF, PetscFree(m[0]));
  PetscCallAbort(PETSC_COMM_SELF, PetscFree(m));
}

static void check_cell(PetscInt order, const PetscReal coords[12], const char *tag) {
  const PetscInt dof = order * (order + 2) * (order + 3) / 2;
  const PetscInt nH1 = (order + 1) * (order + 2) * (order + 3) / 6;

  /* computeElementalMatrices / buildDiscreteGradientMatrix read only fem.order
   * (dof counts are derived internally); the rest is zeroed for cleanliness. */
  FemSpace fem;
  PetscCallAbort(PETSC_COMM_SELF, PetscMemzero(&fem, sizeof(fem)));
  fem.order = order;

  Cell cell;
  PetscCallAbort(PETSC_COMM_SELF, PetscMemzero(&cell, sizeof(cell)));
  for (int i = 0; i < 12; i++) cell.coordinates[i] = coords[i];
  cell.conductivity[0] = cell.conductivity[1] = cell.conductivity[2] = 1.0;

  /* Geometric Jacobian: must be non-degenerate and positively oriented. */
  PetscCallAbort(PETSC_COMM_SELF, computeCellJacobian(&cell));
  PT_CHECK(cell.detJacobian > 1e-12, "[%s] order=%d: detJ=%.3e not positive", tag, (int)order, cell.detJacobian);

  /* Quadrature. */
  Quadrature3D quad;
  PetscCallAbort(PETSC_COMM_SELF, createQuadrature3D(order, &quad));

  PetscReal **Me = alloc2d(dof, dof);
  PetscReal **Ke = alloc2d(dof, dof);
  PetscReal **G  = alloc2d(dof, nH1);

  PetscCallAbort(PETSC_COMM_SELF, computeElementalMatrices(&fem, &cell, &quad, Me, Ke));
  PetscCallAbort(PETSC_COMM_SELF, buildDiscreteGradientMatrix(&fem, &cell, G));

  /* Finiteness + symmetry + scale. */
  PetscReal meMax = 0, keMax = 0;
  int       meSym = 1, keSym = 1, finite = 1;
  for (PetscInt i = 0; i < dof; i++)
    for (PetscInt j = 0; j < dof; j++) {
      if (!isfinite(Me[i][j]) || !isfinite(Ke[i][j])) finite = 0;
      meMax = PetscMax(meMax, fabs(Me[i][j]));
      keMax = PetscMax(keMax, fabs(Ke[i][j]));
      if (fabs(Me[i][j] - Me[j][i]) > 1e-11 * (1.0 + fabs(Me[i][j]))) meSym = 0;
      if (fabs(Ke[i][j] - Ke[j][i]) > 1e-11 * (1.0 + fabs(Ke[i][j]))) keSym = 0;
    }
  PT_CHECK(finite, "[%s] order=%d: non-finite Me/Ke entry", tag, (int)order);
  PT_CHECK(meSym, "[%s] order=%d: Me not symmetric", tag, (int)order);
  PT_CHECK(keSym, "[%s] order=%d: Ke not symmetric", tag, (int)order);
  PT_CHECK(meMax > 0, "[%s] order=%d: Me is all-zero", tag, (int)order);

  /* Definiteness by sampling: x^T Me x > 0 (SPD), x^T Ke x >= 0 (SPSD). */
  PetscReal *x;
  PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(dof, &x));
  unsigned int seed  = 12345u + (unsigned)order;
  int          meSpd = 1, keSpsd = 1;
  for (int i = 0; i < 8; i++) {
    for (PetscInt j = 0; j < dof; j++) {
      seed = seed * 1103515245u + 12345u;
      x[j] = ((seed >> 16) & 0x7fff) / 16383.5 - 1.0;
    }
    PetscReal qm = 0, qk = 0;
    for (PetscInt j = 0; j < dof; j++)
      for (PetscInt k = 0; k < dof; k++) {
        qm += x[j] * Me[j][k] * x[k];
        qk += x[j] * Ke[j][k] * x[k];
      }
    if (qm <= 1e-13 * meMax) meSpd = 0;
    if (qk < -1e-9 * (keMax + 1.0)) keSpsd = 0;
  }
  PT_CHECK(meSpd, "[%s] order=%d: Me not positive definite (sampled)", tag, (int)order);
  PT_CHECK(keSpsd, "[%s] order=%d: Ke not positive semidefinite (sampled)", tag, (int)order);
  PetscCallAbort(PETSC_COMM_SELF, PetscFree(x));

  /* De Rham: Ke . G == 0 (columns of G span the curl-kernel). */
  PetscReal kgMax = 0;
  for (PetscInt i = 0; i < dof; i++)
    for (PetscInt j = 0; j < nH1; j++) {
      PetscReal v = 0;
      for (PetscInt k = 0; k < dof; k++) v += Ke[i][k] * G[k][j];
      kgMax = PetscMax(kgMax, fabs(v));
    }
  PT_CHECK(kgMax < 1e-8 * (keMax + 1.0), "[%s] order=%d: ||Ke.G||_max = %.3e (expected ~0, keMax=%.3e)", tag, (int)order, kgMax, keMax);

  /* tetrahedronXyzToReference(v0 + J^T r) == r */
  const PetscReal refPoints[4][3] = {{0.25, 0.25, 0.25}, {0.1, 0.2, 0.3}, {0.6, 0.1, 0.05}, {0.0, 0.3, 0.3}};
  for (int i = 0; i < 4; i++) {
    PetscReal x[3], r[3];
    for (int d = 0; d < 3; d++)
      x[d] = coords[d] + refPoints[i][0] * (coords[3 + d] - coords[d]) + refPoints[i][1] * (coords[6 + d] - coords[d]) +
             refPoints[i][2] * (coords[9 + d] - coords[d]);
    PetscCallAbort(PETSC_COMM_SELF, tetrahedronXyzToReference(coords, x, r));
    for (int d = 0; d < 3; d++)
      PT_CLOSE(r[d], refPoints[i][d], 1e-12, "[%s] order=%d: tetrahedronXyzToReference point %d component %d = %.15g, expected %.15g", tag,
               (int)order, i, d, r[d], refPoints[i][d]);
  }

  check_interpolation(order, coords, G, tag);

  free2d(Me);
  free2d(Ke);
  free2d(G);
  PetscCallAbort(PETSC_COMM_SELF, destroyQuadrature3D(&quad));
}

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscInt order = (argc > 1) ? atoi(argv[1]) : 1;

  const PetscReal refTet[12]  = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const PetscReal skewTet[12] = {0.2, 0.1, -0.3, 1.4, 0.0, 0.1, 0.3, 1.7, 0.2, -0.1, 0.4, 2.1};

  check_cell(order, refTet, "reference");
  check_cell(order, skewTet, "skewed");

  int rc = pt_report("test_elements");
  PetscCall(PetscFinalize());
  return rc;
}
