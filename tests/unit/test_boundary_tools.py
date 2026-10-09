"""Boundary tools: 2D triangle quadrature and outer-boundary faces (all orders 1..6).

Drives tests/csrc/test_boundary.c on a distributed box mesh and checks:
  * the 2D rule: weights sum to 1/2, exact monomials up to degree 2*order+1,
  * setupNedelecGrid DOF counts for PETGEM_BC_NATURAL and PETGEM_BC_PEC,
  * getBoundaryFaces / computeBoundaryFaceGeometry: face count, total area,
    sum of n*area = 0, axis-aligned outward normals.
"""
import pytest

from fmcsem_testlib import ORDERS, run_harness


@pytest.mark.parametrize("nprocs", [1, 3])
@pytest.mark.parametrize("order", ORDERS)
def test_boundary_tools(harnesses, order, nprocs):
    r = run_harness(harnesses["test_boundary"], order, nprocs)
    assert r.returncode == 0, (
        f"boundary harness failed at order {order}, {nprocs} ranks\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert "0 failures" in r.stdout
