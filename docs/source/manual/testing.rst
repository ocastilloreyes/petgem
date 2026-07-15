=======
Testing
=======

**PETGEM** ships a `pytest <https://docs.pytest.org/>`_ suite under ``tests/``,
covering the **forward** kernel (``fm.csem``). The inverse kernel is out of
scope for the suite. The full rationale is in ``tests/README.md``.

The tests never modify the production sources: the C harnesses link against the
unchanged ``src/`` translation units, and the end-to-end tests drive the built
binary.

Structure
---------
Six levels of increasing scope, split so the finite-element core is checked at
**every** order cheaply, while the expensive full-pipeline runs are limited to
the orders that exercise new code paths.

.. list-table::
   :header-rows: 1
   :widths: 8 34 14 44

   * - Level
     - What it verifies
     - Orders
     - How
   * - 1
     - Basis functions
     - 1-6
     - C harness against ``fe_nedelec.c``, ``fe_nodal.c``
   * - 2
     - DOF ordering
     - 1-6
     - C harness against ``fe_nedelec.c``, ``fe_nodal.c``
   * - 3
     - Element matrices
     - 1-6
     - C harness against ``fem.c``
   * - 4
     - Global assembly
     - 1-3
     - ``fm.csem`` (assembly only)
   * - 5
     - Basic physics
     - 1-3
     - ``fm.csem`` with a direct LU/MUMPS solve
   * - 6
     - Order of accuracy (MMS)
     - 1, 3, 6
     - ``fm.csem -mms`` vs. committed golden errors

Levels 1-3 (``tests/unit/``) are fast and need PETSc and ``mpicc``, but no
built binary. Levels 4-5 (``tests/e2e/``) run the assemble → solve →
interpolate pipeline on ``examples/unit_cube``. Orders 1, 2, 3 are enough there
because they activate every DOF entity class - order 1 has edge DOFs only,
order 2 adds face DOFs, order 3 adds interior DOFs; higher orders only add more
DOFs of the same classes, already covered by levels 1-3.

Level 6 (``tests/mms/``) is the order-of-accuracy check. It runs ``fm.csem
-mms`` on the :math:`[0,1]^3` cube with a manufactured solution (see
:doc:`method`), and compares the reported relative :math:`L^2` and
:math:`H(\mathrm{curl})` errors and the DOF count against
``tests/mms/reference/mms_golden.json``. Its bundle is self-contained and built
on demand by ``tests/mms/make_bundle.sh``.

Markers, declared in ``pytest.ini``:

- ``e2e`` - needs the built binary and the ``unit_cube`` dataset; skipped
  automatically when the binary is unavailable.
- ``slow`` - the full-pipeline family (levels 4-5).

``testpaths`` is ``tests/unit tests/e2e``, so a bare ``pytest`` does not collect
level 6; run it explicitly.

Running
-------
.. code-block:: bash

   # Levels 1-3 - all orders 1..6, fast (needs PETSc + mpicc; no binary)
   pytest tests/unit

   # Levels 4-5 - orders 1..3 (needs the fm.csem binary)
   PETGEM_FM_CSEM=build/fm.csem pytest tests/e2e

   # ... with 4 MPI tasks for the solves
   PETGEM_FM_CSEM=build/fm.csem FM_CSEM_NP=4 pytest tests/e2e

   # Level 6 - MMS (needs the binary + gmsh; build the N=4 bundle first)
   bash tests/mms/make_bundle.sh 4
   PETGEM_FM_CSEM=build/fm.csem FM_CSEM_NP=2 pytest tests/mms

Environment variables:

- ``PETGEM_FM_CSEM`` - path to the ``fm.csem`` binary. Without it, the tests
  that need a binary skip.
- ``FM_CSEM_NP`` - number of MPI ranks for the kernel runs.

Fixtures skip cleanly when the PETSc toolchain (levels 1-3) or the binary
(levels 4-6) is unavailable, so a partial environment still runs whatever it
can.

Determinism and tolerances
--------------------------
- Levels 1-3 assert analytic invariants (DOF-count closed forms, H(curl)
  constant-vector reproduction, partition of unity, symmetry of the element
  matrices, and the de Rham identity :math:`K_e G_e = 0`) to about ``1e-9``.
- Level 5 compares a **direct LU (MUMPS)** solve against committed exact-LU
  golden responses to a relative tolerance of ``1e-3``. A direct solve is
  deterministic and always succeeds, so the check is stable across rank counts
  and does not depend on iterative-solver convergence.
- Level 6 compares the MMS error norms against the golden with a combined
  ``atol + rtol*|golden|`` tolerance (``rtol=1e-3``, ``atol=1e-8``), and
  requires the backward residual :math:`\|Ax-b\|/\|b\|` to stay below
  ``1e-10``.

Golden references are exact serial LU solves of the current code; regenerate
them only after an intentional change to the forward result. The commands are in
``tests/README.md`` and ``examples/unit_cube/README.md``.

Continuous integration
----------------------
``.github/workflows/ci-develop.yml`` compiles ``fm.csem`` and calls the reusable
``tests-fm-csem.yml`` workflow, which runs the suite (levels 1-3; levels 4-5 and
6 against the compiled binary) plus an Extrae smoke run, inside the project's CI
image. Code and documentation jobs are gated independently, so a docs-only
change skips the kernel build.
