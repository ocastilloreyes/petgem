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
 *   - a positively oriented, non-degenerate cell Jacobian.
 *
 * Usage: test_elements <order>   (order in 1..6).  Exit 0 iff all checks pass.
 */
#include "fem.h"
#include "grid.h"
#include "constants.h"
#include "petgem_test.h"
#include <petsc.h>

static PetscReal **alloc2d(PetscInt r, PetscInt c)
{
  PetscReal **m, *flat;
  PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(r, &m));
  PetscCallAbort(PETSC_COMM_SELF, PetscCalloc1(r * c, &flat));
  for (PetscInt i = 0; i < r; i++) m[i] = flat + i * c;
  return m;
}
static void free2d(PetscReal **m) { PetscCallAbort(PETSC_COMM_SELF, PetscFree(m[0])); PetscCallAbort(PETSC_COMM_SELF, PetscFree(m)); }

static void check_cell(PetscInt order, const PetscReal coords[12], const char *tag)
{
  const PetscInt dof = order * (order + 2) * (order + 3) / 2;
  const PetscInt nH1 = (order + 1) * (order + 2) * (order + 3) / 6;

  /* computeElementalMatrices / buildDiscreteGradientMatrix read only fem.order
   * (dof counts are derived internally); the rest is zeroed for cleanliness. */
  FEMSpace fem;
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
  PetscCallAbort(PETSC_COMM_SELF, computeNum3DQuadraturePoints(order, &quad));
  PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(quad.numPoints, &quad.points));
  for (PetscInt q = 0; q < quad.numPoints; q++)
    PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(NUM_DIMENSIONS, &quad.points[q]));
  PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(quad.numPoints, &quad.weights));
  PetscCallAbort(PETSC_COMM_SELF, compute3DQuadraturePoints(&quad));

  PetscReal **Me = alloc2d(dof, dof);
  PetscReal **Ke = alloc2d(dof, dof);
  PetscReal **G  = alloc2d(dof, nH1);

  PetscCallAbort(PETSC_COMM_SELF, computeElementalMatrices(&fem, &cell, &quad, Me, Ke));
  PetscCallAbort(PETSC_COMM_SELF, buildDiscreteGradientMatrix(&fem, &cell, G));

  /* Finiteness + symmetry + scale. */
  PetscReal meMax = 0, keMax = 0;
  int meSym = 1, keSym = 1, finite = 1;
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
  unsigned int seed = 12345u + (unsigned)order;
  int meSPD = 1, keSPSD = 1;
  for (int s = 0; s < 8; s++) {
    for (PetscInt i = 0; i < dof; i++) { seed = seed*1103515245u + 12345u; x[i] = ((seed >> 16) & 0x7fff) / 16383.5 - 1.0; }
    PetscReal qm = 0, qk = 0;
    for (PetscInt i = 0; i < dof; i++)
      for (PetscInt j = 0; j < dof; j++) { qm += x[i]*Me[i][j]*x[j]; qk += x[i]*Ke[i][j]*x[j]; }
    if (qm <= 1e-13 * meMax) meSPD = 0;
    if (qk < -1e-9 * (keMax + 1.0)) keSPSD = 0;
  }
  PT_CHECK(meSPD, "[%s] order=%d: Me not positive definite (sampled)", tag, (int)order);
  PT_CHECK(keSPSD, "[%s] order=%d: Ke not positive semidefinite (sampled)", tag, (int)order);
  PetscCallAbort(PETSC_COMM_SELF, PetscFree(x));

  /* De Rham: Ke . G == 0 (columns of G span the curl-kernel). */
  PetscReal kgMax = 0;
  for (PetscInt i = 0; i < dof; i++)
    for (PetscInt j = 0; j < nH1; j++) {
      PetscReal v = 0;
      for (PetscInt k = 0; k < dof; k++) v += Ke[i][k] * G[k][j];
      kgMax = PetscMax(kgMax, fabs(v));
    }
  PT_CHECK(kgMax < 1e-8 * (keMax + 1.0), "[%s] order=%d: ||Ke.G||_max = %.3e (expected ~0, keMax=%.3e)",
           tag, (int)order, kgMax, keMax);

  free2d(Me); free2d(Ke); free2d(G);
  for (PetscInt q = 0; q < quad.numPoints; q++) PetscCallAbort(PETSC_COMM_SELF, PetscFree(quad.points[q]));
  PetscCallAbort(PETSC_COMM_SELF, PetscFree(quad.points));
  PetscCallAbort(PETSC_COMM_SELF, PetscFree(quad.weights));
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscInt order = (argc > 1) ? atoi(argv[1]) : 1;

  const PetscReal refTet[12]   = {0,0,0, 1,0,0, 0,1,0, 0,0,1};
  const PetscReal skewTet[12]  = {0.2,0.1,-0.3, 1.4,0.0,0.1, 0.3,1.7,0.2, -0.1,0.4,2.1};

  check_cell(order, refTet,  "reference");
  check_cell(order, skewTet, "skewed");

  int rc = pt_report("test_elements");
  PetscCall(PetscFinalize());
  return rc;
}
