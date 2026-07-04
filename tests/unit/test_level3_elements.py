"""LEVEL 3 - Element-level matrix tests (all orders 1..6).

Drives tests/csrc/test_elements.c, which builds Me / Ke / G through the real
production entry points on a reference and a skewed cell and checks:
  * finiteness, symmetry of Me and Ke,
  * positive-definiteness of Me and positive-semidefiniteness of Ke (sampled),
  * the De Rham identity Ke . G == 0 (discrete gradient in the curl kernel),
  * a positively oriented, non-degenerate cell Jacobian.
"""
import pytest

from fmcsem_testlib import ORDERS, run_harness


@pytest.mark.parametrize("order", ORDERS)
def test_elemental_matrices(harnesses, order):
    r = run_harness(harnesses["test_elements"], order)
    assert r.returncode == 0, (
        f"elements harness failed at order {order}\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert "0 failures" in r.stdout
