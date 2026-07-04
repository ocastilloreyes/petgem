/*
 * Filename: test_basis.c
 * Author: PETGEM test suite
 * Date: 2026-07-03
 *
 * Description:
 * LEVEL 1 - Basis function tests. Exercises the unchanged reference-element
 * bases (fe_nedelec.c H(curl), fe_nodal.c H1) for a given order (argv[1]):
 *   - DOF-count closed forms (expected polynomial-order behaviour);
 *   - H(curl) constant-vector reproduction (the Nedelec partition-of-unity:
 *     sum_m (c . tangent_m) phi_m(x) == c for every constant field c);
 *   - H1 partition of unity (sum_j phi_j == 1) and gradient consistency
 *     (sum_j grad phi_j == 0), the P0/P1 reproduction invariants.
 *
 * Usage: test_basis <order>   (order in 1..6).  Exit 0 iff all checks pass.
 */
#include "fe_nedelec.h"
#include "fe_nodal.h"
#include "constants.h"
#include "petgem_test.h"
#include <petsc.h>

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscInt order = (argc > 1) ? atoi(argv[1]) : 1;

  /* Reference-cell sample points (interior of the unit tetrahedron). */
  const PetscReal pts[5][3] = {
    {0.10, 0.20, 0.30}, {0.30, 0.05, 0.40}, {0.25, 0.25, 0.25},
    {0.40, 0.35, 0.05}, {0.05, 0.10, 0.15}
  };

  /* ---------- Nedelec H(curl) ---------- */
  FeNedelec *fe;
  PetscCall(feNedelecCreate(order, &fe));
  PetscInt dof = feNedelecGetDof(fe);

  PT_CHECK(dof == order * (order + 2) * (order + 3) / 2,
           "Nedelec dof count: order=%d got=%d expected=%d",
           (int)order, (int)dof, (int)(order * (order + 2) * (order + 3) / 2));
  PT_CHECK(feNedelecGetOrder(fe) == order, "Nedelec stored order mismatch");
  PT_CHECK(feNedelecDofCount(order) == dof, "feNedelecDofCount disagrees with handle");

  /* Cache each DOF's defining tangent (used by the reproduction functional). */
  PetscReal *tang;
  PetscCall(PetscMalloc1(dof * 3, &tang));
  for (PetscInt m = 0; m < dof; m++) {
    PetscReal node[3], tg[3];
    PetscCall(feNedelecGetDofInfo(fe, m, node, tg));
    tang[m*3+0] = tg[0]; tang[m*3+1] = tg[1]; tang[m*3+2] = tg[2];
  }

  /* Constant-vector reproduction: the lowest polynomial the H(curl) space must
   * reproduce exactly is any constant field c. With the nodal (dual) basis the
   * interpolant of c is sum_m (c.tangent_m) phi_m == c. */
  const PetscReal cs[3][3] = {{1,0,0}, {0,1,0}, {0.7,-1.3,0.9}};
  PetscReal *shape;
  PetscCall(PetscMalloc1(dof * 3, &shape));
  for (int q = 0; q < 5; q++) {
    PetscCall(feNedelecCalcVShape(fe, pts[q][0], pts[q][1], pts[q][2], shape));
    for (int ci = 0; ci < 3; ci++) {
      PetscReal S[3] = {0, 0, 0};
      for (PetscInt m = 0; m < dof; m++) {
        PetscReal cdott = cs[ci][0]*tang[m*3] + cs[ci][1]*tang[m*3+1] + cs[ci][2]*tang[m*3+2];
        for (int d = 0; d < 3; d++) S[d] += cdott * shape[m*3+d];
      }
      for (int d = 0; d < 3; d++)
        PT_CHECK(fabs(S[d] - cs[ci][d]) < 1e-9,
                 "Nedelec const reproduction order=%d q=%d c=%d comp=%d got=%.3e exp=%.3e",
                 (int)order, q, ci, d, S[d], cs[ci][d]);
    }
  }
  PetscCall(PetscFree(shape));
  PetscCall(PetscFree(tang));
  PetscCall(feNedelecDestroy(&fe));

  /* ---------- Nodal H1 ---------- */
  PT_CHECK(feNodalSupports(order), "feNodalSupports(%d) is false", (int)order);
  if (feNodalSupports(order)) {
    const PetscInt nH1 = (order+1)*(order+2)*(order+3)/6;
    const PetscReal vc[NUM_VERTICES_PER_CELL*NUM_DIMENSIONS] = {0,0,0, 1,0,0, 0,1,0, 0,0,1};
    PetscReal *ShapH, *Gbuf;
    PetscCall(PetscMalloc1(nH1, &ShapH));
    PetscCall(PetscMalloc1(3*nH1, &Gbuf));
    PetscReal *GradH[3] = {Gbuf, Gbuf + nH1, Gbuf + 2*nH1};
    for (int q = 0; q < 5; q++) {
      PetscCall(feNodalH1Shape(order, vc, pts[q], ShapH, GradH));
      PetscReal sum = 0.0;
      for (PetscInt j = 0; j < nH1; j++) sum += ShapH[j];
      PT_CHECK(fabs(sum - 1.0) < 1e-10, "H1 partition of unity order=%d q=%d sum=%.3e", (int)order, q, sum);
      for (int d = 0; d < 3; d++) {
        PetscReal gs = 0.0;
        for (PetscInt j = 0; j < nH1; j++) gs += GradH[d][j];
        PT_CHECK(fabs(gs) < 1e-9, "H1 gradient-sum order=%d q=%d dim=%d got=%.3e", (int)order, q, d, gs);
      }
    }
    PetscCall(PetscFree(ShapH));
    PetscCall(PetscFree(Gbuf));
  }

  int rc = pt_report("test_basis");
  PetscCall(PetscFinalize());
  return rc;
}
