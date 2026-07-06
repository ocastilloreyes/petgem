"""LEVEL 4 - Global assembly tests (unit cube, pipeline orders 1..3).

Runs fm.csem end-to-end with a trivial solver (assembly is the object under
test, the "solution" is ignored) and checks, from the real assembled system:
  * the per-cell discrete-gradient kernel property held everywhere during
    assembly (no `Error element` from checkGradientKernel),
  * the mesh topology loaded from the bundle is the expected, order-independent
    unit-cube topology,
  * per-order Nedelec DOF-per-entity counts follow the closed forms,
  * the global operator is square, consistent with the RHS vector size, and
    bounded above by the total (pre-boundary-condition) DOF count.

Orders 1/2/3 exercise every DOF entity class (edge / face / interior), so the
assembly code paths are fully covered here; the FE core is verified for ALL
orders 1..6 by the fast level 1-3 harnesses.
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
    for k, v in MESH.items():
        assert s.get(k) == v, f"order {order}: mesh {k}={s.get(k)} expected {v}"

    # (c) per-order Nedelec DOF-per-entity closed forms.
    assert s["order"] == order
    assert s["dof_vertex"] == 0
    assert s["dof_edge"] == order
    assert s["dof_face"] == order * (order - 1)
    assert s["dof_volume"] == order * (order - 1) * (order - 2) // 2

    # (d) global operator: square, matches the RHS layout, bounded by total DOFs.
    assert s["matrix_rows"] == s["vector_size"] > 0, "operator not square / vector-size mismatch"
    total = (s["edges"] * s["dof_edge"] + s["faces"] * s["dof_face"] + s["cells"] * s["dof_volume"])
    assert 0 < s["matrix_rows"] <= total, (
        f"order {order}: free DOFs {s['matrix_rows']} not in (0, total {total}] "
        f"(boundary DOFs are constrained, so free <= total)")
