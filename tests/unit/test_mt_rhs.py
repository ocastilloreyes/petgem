"""MT boundary right-hand side, Eq. (10) (assembleMtBoundaryRHS, all orders 1..6).

Drives tests/csrc/test_mt_rhs.c and checks, for a field F in the Nedelec
space with coefficients c from its L2 projection,
    c^T b = -iωμ ∮ F . (n x Ĥ) dΓ
against the closed form on a box with H(z) linear, for both polarizations,
plus the size of B and the rejection of a PEC grid.
"""
import pytest

from fmcsem_testlib import ORDERS, run_harness


@pytest.mark.parametrize("nprocs", [1, 3])
@pytest.mark.parametrize("order", ORDERS)
def test_mt_rhs(harnesses, order, nprocs):
    r = run_harness(harnesses["test_mt_rhs"], order, nprocs)
    assert r.returncode == 0, (
        f"MT RHS harness failed at order {order}, {nprocs} ranks\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert "0 failures" in r.stdout
