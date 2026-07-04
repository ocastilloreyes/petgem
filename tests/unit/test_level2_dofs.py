"""LEVEL 2 - DOF ordering / enumeration tests (all orders 1..6).

Drives tests/csrc/test_dofs.c, which verifies the Nedelec local DOF layout:
  * per-entity DOF partition (edges / faces / interior) matches the closed
    forms, classified geometrically from each DOF node,
  * defining nodes lie inside the reference tet and tangents are valid,
  * the edge-DOF block comes first (native ordering) - a regression guard on
    the enumeration used by the assembly's local-to-global mapping.
"""
import pytest

from fmcsem_testlib import ORDERS, run_harness


@pytest.mark.parametrize("order", ORDERS)
def test_dof_ordering(harnesses, order):
    r = run_harness(harnesses["test_dofs"], order)
    assert r.returncode == 0, (
        f"dof harness failed at order {order}\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert "0 failures" in r.stdout
