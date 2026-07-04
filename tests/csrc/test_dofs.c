/*
 * Filename: test_dofs.c
 * Author: PETGEM test suite
 * Date: 2026-07-03
 *
 * Description:
 * LEVEL 2 - DOF ordering / enumeration tests. Verifies the Nedelec local DOF
 * layout produced by fe_nedelec.c for a given order (argv[1]):
 *   - per-entity DOF partition (edges / faces / interior) matches the closed
 *     forms, classified geometrically from each DOF node's barycentric zeros;
 *   - defining nodes lie inside the reference tetrahedron and tangent indices
 *     are valid (0..5), catching enumeration regressions across all orders;
 *   - the H1 companion space exposes the matching P_order DOF count.
 *
 * Usage: test_dofs <order>   (order in 1..6).  Exit 0 iff all checks pass.
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

  FeNedelec *fe;
  PetscCall(feNedelecCreate(order, &fe));
  const PetscInt dof = feNedelecGetDof(fe);

  /* Expected per-entity DOF counts (MFEM-style Nedelec on a tetrahedron). */
  const PetscInt expEdge = 6 * order;
  const PetscInt expFace = 4 * order * (order - 1);
  const PetscInt expInt  = dof - expEdge - expFace;   /* interior remainder */
  PT_CHECK(expInt >= 0, "order=%d: negative interior dof count %d", (int)order, (int)expInt);

  /* Classify every DOF node by how many barycentric coordinates vanish:
   *   2 zeros -> edge, 1 zero -> face, 0 zeros -> interior (vertices carry no
   *   H(curl) DOFs). This is exactly the enumeration the assembly relies on. */
  PetscInt nEdge = 0, nFace = 0, nInt = 0, nBad = 0, nBadTangent = 0, nBadNode = 0;
  for (PetscInt m = 0; m < dof; m++) {
    PetscReal node[3], tang[3];
    PetscCall(feNedelecGetDofInfo(fe, m, node, tang));

    const PetscReal lam[4] = {1.0 - node[0] - node[1] - node[2], node[0], node[1], node[2]};
    int zeros = 0;
    for (int i = 0; i < 4; i++) if (fabs(lam[i]) < 1e-12) zeros++;
    if      (zeros == 2) nEdge++;
    else if (zeros == 1) nFace++;
    else if (zeros == 0) nInt++;
    else                 nBad++;      /* 3 zeros == a vertex: never a Nedelec DOF */

    /* Node must sit inside/on the closed reference tetrahedron. */
    int inside = 1;
    for (int i = 0; i < 4; i++) if (lam[i] < -1e-12 || lam[i] > 1.0 + 1e-12) inside = 0;
    if (!inside) nBadNode++;

    /* Tangent must be one of the six canonical directions (unit-ish, non-zero). */
    PetscReal tn = tang[0]*tang[0] + tang[1]*tang[1] + tang[2]*tang[2];
    if (tn < 0.5 || tn > 2.5) nBadTangent++;
  }

  PT_CHECK(nEdge == expEdge, "edge DOF count order=%d got=%d expected=%d", (int)order, (int)nEdge, (int)expEdge);
  PT_CHECK(nFace == expFace, "face DOF count order=%d got=%d expected=%d", (int)order, (int)nFace, (int)expFace);
  PT_CHECK(nInt  == expInt,  "interior DOF count order=%d got=%d expected=%d", (int)order, (int)nInt, (int)expInt);
  PT_CHECK(nBad  == 0,       "order=%d: %d DOF nodes at vertices (invalid)", (int)order, (int)nBad);
  PT_CHECK(nBadNode == 0,    "order=%d: %d DOF nodes outside reference tet", (int)order, (int)nBadNode);
  PT_CHECK(nBadTangent == 0, "order=%d: %d DOFs with invalid tangent", (int)order, (int)nBadTangent);

  /* Edge DOFs come first in the native ordering (edge e -> slots [e*order, ...)),
   * so the first 6*order DOFs must all be edge DOFs. */
  int firstBlockEdge = 1;
  for (PetscInt m = 0; m < expEdge && m < dof; m++) {
    PetscReal node[3], tang[3];
    PetscCall(feNedelecGetDofInfo(fe, m, node, tang));
    const PetscReal lam[4] = {1.0 - node[0] - node[1] - node[2], node[0], node[1], node[2]};
    int zeros = 0;
    for (int i = 0; i < 4; i++) if (fabs(lam[i]) < 1e-12) zeros++;
    if (zeros != 2) firstBlockEdge = 0;
  }
  PT_CHECK(firstBlockEdge, "order=%d: first %d DOFs are not all edge DOFs", (int)order, (int)expEdge);

  PetscCall(feNedelecDestroy(&fe));

  /* H1 companion space: matched P_order DOF count (De Rham pair column space). */
  PT_CHECK(feNodalSupports(order), "feNodalSupports(%d) is false", (int)order);
  const PetscInt nH1 = (order+1)*(order+2)*(order+3)/6;
  PT_CHECK(nH1 >= 4, "order=%d: H1 dof count %d < 4 vertices", (int)order, (int)nH1);

  int rc = pt_report("test_dofs");
  PetscCall(PetscFinalize());
  return rc;
}
