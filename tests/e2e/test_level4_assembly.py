"""LEVEL 4 - Global assembly (unit cube, orders 1..3).

Runs fm.csem with a trivial solver and checks, from the run report:
  * no `Error element` from the per-cell gradient-kernel check,
  * the unit-cube mesh topology,
  * the Nedelec DOF-per-entity closed forms,
  * 0 < global DOFs <= total DOFs (boundary DOFs are constrained).
"""
import pytest

from fmcsem_testlib import PIPELINE_ORDERS, parse_grid_stats

# The whole e2e family is the "slow" bucket (full pipeline on the fixed mesh).
pytestmark = pytest.mark.slow

# Fixed unit-cube topology (order-independent) - a regression guard on loading.
# Matches the structured mesh.geo (N=6): (N+1)^3 vertices, 6*N^3 tetrahedra.
MESH = {"vertices": 343, "edges": 1854, "faces": 2808, "cells": 1296}


@pytest.mark.e2e
@pytest.mark.parametrize("order", PIPELINE_ORDERS)
def test_global_assembly(fm_run, order):
    run = fm_run(order, "assemble")
    assert run.returncode == 0, f"fm.csem assembly failed (order {order}):\n{run.stdout[-3000:]}"

    # (a) De Rham kernel property held on every cell during assembly.
    assert "Error element" not in run.stdout, f"checkGradientKernel: K_e.G_e != 0 at order {order}"

    s = parse_grid_stats(run.stdout)

    # (b) mesh topology is order-independent and fixed for this dataset.
    for key, expected in MESH.items():
        assert s.get(key) == expected, f"order {order}: mesh {key}={s.get(key)} expected {expected}"

    # (c) per-order Nedelec DOF-per-entity closed forms.
    assert s["order"] == order
    assert s["dof_vertex"] == 0
    assert s["dof_edge"] == order
    assert s["dof_face"] == order * (order - 1)
    assert s["dof_volume"] == order * (order - 1) * (order - 2) // 2

    # (d) global DOF count bounded by the total (pre-boundary-condition) count.
    total = (s["edges"] * s["dof_edge"] + s["faces"] * s["dof_face"] + s["cells"] * s["dof_volume"])
    assert 0 < s["dofs"] <= total, (
        f"order {order}: free DOFs {s['dofs']} not in (0, total {total}] "
        f"(boundary DOFs are constrained, so free <= total)")
