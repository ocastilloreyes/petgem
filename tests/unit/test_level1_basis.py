"""LEVEL 1 - Basis function tests (all orders 1..6).

Drives the C harness tests/csrc/test_basis.c, which links the unchanged
reference-element bases and checks, at reference-cell points:
  * H(curl) and H1 DOF-count closed forms (expected polynomial-order behaviour),
  * Nedelec constant-vector reproduction (H(curl) partition of unity),
  * H1 partition of unity and gradient-sum-zero.
"""
import pytest

from fmcsem_testlib import ORDERS, run_harness


@pytest.mark.parametrize("order", ORDERS)
def test_basis_functions(harnesses, order):
    r = run_harness(harnesses["test_basis"], order)
    assert r.returncode == 0, (
        f"basis harness failed at order {order}\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert "0 failures" in r.stdout
