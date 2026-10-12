"""C harnesses under tests/csrc/, linked against the production sources.

    level 1   test_basis     Nedelec/H1 DOF counts, constant-vector reproduction,
                             H1 partition of unity and gradient sum
    level 2   test_dofs      per-entity DOF partition, node/tangent validity,
                             edge-first ordering
    level 3   test_elements  Me/Ke symmetry and definiteness, Ke . G = 0,
                             cell Jacobian
    boundary  test_boundary  2D quadrature, boundary faces, setupNedelecGrid
                             with PETGEM_BC_NATURAL and PETGEM_BC_PEC
    MT 1D     test_mt1d      solveMt1D vs the exact layered solution,
                             evalMt1DField, classifyMtBoxFaces, buildMt1DProfile
    MT RHS    test_mt_rhs    assembleMtBoundaryRhs against Eq. (10)

Each harness prints "<n> failures" and exits non-zero on failure.
"""
import pytest

from fmcsem_testlib import ORDERS, run_harness

# (harness, orders, rank counts)
CASES = [
    ("test_basis",    ORDERS, [1]),
    ("test_dofs",     ORDERS, [1]),
    ("test_elements", ORDERS, [1]),
    ("test_boundary", ORDERS, [1, 3]),
    ("test_mt1d",     [1],    [1, 3]),
    ("test_mt_rhs",   ORDERS, [1, 3]),
]

PARAMS = [pytest.param(name, order, nprocs, id=f"{name}-p{order}-np{nprocs}")
          for name, orders, ranks in CASES for order in orders for nprocs in ranks]


@pytest.mark.parametrize("name, order, nprocs", PARAMS)
def test_harness(harnesses, name, order, nprocs):
    r = run_harness(harnesses[name], order, nprocs)
    assert r.returncode == 0, (
        f"{name} failed at order {order}, {nprocs} ranks\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert ", 0 failures" in r.stdout
