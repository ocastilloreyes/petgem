"""Layered architecture of src/ and include/.

A module may include headers of its own layer or of a lower one. Every module
must be listed here, so a new file has to be placed in a layer explicitly.

    0  base           constants, common, version
    1  generic I/O    io
    2  discretization fe_nedelec, fe_nodal, grid, fem
    3  numerics       assembly, solver, receiver_interp
    4  physics        csem, mt
    5  inversion      inversion, inversion_internal, inversion_io,
                      inversion_smoother, lbfgs
    6  application    runtime, kernels, fm_csem, fm_mt, im_csem, petgem, *_main
"""
import re

import pytest

from fmcsem_testlib import REPO_ROOT

LAYERS = [
    ["constants", "common", "version"],
    ["io"],
    ["fe_nedelec", "fe_nodal", "grid", "fem"],
    ["assembly", "solver", "receiver_interp"],
    ["csem", "mt"],
    ["inversion", "inversion_internal", "inversion_io", "inversion_smoother", "lbfgs"],
    ["runtime", "kernels", "fm_csem", "fm_mt", "im_csem", "petgem",
     "fm_csem_main", "fm_mt_main", "im_csem_main"],
]
LAYER = {module: level for level, modules in enumerate(LAYERS) for module in modules}

# Generated or third-party headers, not part of the layering.
EXTERNAL = {"git_rev", "extrae_user_events"}

# Physics modules that must stay independent of each other.
INDEPENDENT = [("csem", "mt"), ("mt", "csem")]

SOURCES = sorted((REPO_ROOT / "src").glob("*.c")) + sorted((REPO_ROOT / "include").glob("*.h"))
INCLUDE = re.compile(r'^\s*#include\s+"([A-Za-z0-9_]+)\.h"', re.M)


def includes(path):
    return [h for h in INCLUDE.findall(path.read_text()) if h not in EXTERNAL]


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_module_has_a_layer(path):
    assert path.stem in LAYER, f"{path.name}: add '{path.stem}' to a layer in {__file__}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_upward_dependency(path):
    level = LAYER.get(path.stem)
    if level is None:
        pytest.skip("module without a layer (see test_module_has_a_layer)")
    upward = [h for h in includes(path) if LAYER.get(h, -1) > level]
    assert not upward, (f"{path.name} (layer {level}) includes higher layers: "
                        + ", ".join(f"{h}.h (layer {LAYER[h]})" for h in upward))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_includes_are_known(path):
    unknown = [h for h in includes(path) if h not in LAYER]
    assert not unknown, f"{path.name} includes headers without a layer: {unknown}"


@pytest.mark.parametrize("a, b", INDEPENDENT)
def test_physics_modules_are_independent(a, b):
    for path in (REPO_ROOT / "src" / f"{a}.c", REPO_ROOT / "include" / f"{a}.h"):
        assert b not in includes(path), f"{path.name} must not depend on {b}"
