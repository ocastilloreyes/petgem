=======
Testing
=======

The `pytest <https://docs.pytest.org/>`_ suite under ``tests/`` covers the
three kernels. The C harnesses link against the unchanged ``src/`` files, and
the end-to-end tests run the built binaries. ``tests/README.md`` lists the
files and the exact checks.

Test levels
-----------
.. list-table::
   :header-rows: 1
   :widths: 10 30 12 48

   * - Level
     - What it checks
     - Orders
     - How
   * - 1
     - Basis functions
     - 1-6
     - C harness ``test_basis.c``
   * - 2
     - DOF ordering
     - 1-6
     - C harness ``test_dofs.c``
   * - 3
     - Element matrices
     - 1-6
     - C harness ``test_elements.c``
   * - 4
     - Global assembly
     - 1-3
     - ``fm.csem`` on ``examples/fm_csem_unit_cube``
   * - 5
     - Fields at the receivers
     - 1-3
     - ``fm.csem`` with MUMPS against golden responses
   * - 6
     - Order of accuracy (MMS)
     - 1, 3, 6
     - ``build/mms`` against golden error norms
   * - MT
     - Boundary quadrature, 1D field, boundary right-hand side
     - 1-6
     - C harnesses ``test_boundary.c``, ``test_mt1d.c``, ``test_mt_rhs.c``
   * - MT
     - Half-space response
     - 2
     - ``fm.mt`` on 1 and 3 ranks
   * - IM
     - Reduced ``im_csem_block`` inversion
     - 1
     - ``im.csem``, 2 L-BFGS iterations against a golden

Levels 4 and 5 stop at order 3: orders 1, 2 and 3 already have edge, face and
interior DOFs, and higher orders only add more of the same. Level 6 runs the
manufactured-solution program described in :doc:`method`.

``tests/unit/test_layers.py`` checks the module layering described in
:doc:`contributing`.

Running
-------
.. code-block:: bash

   # Levels 1-3 and the MT harnesses (needs PETSc and mpicc)
   pytest tests/unit

   # Levels 4-5
   bash examples/fm_csem_unit_cube/scripts/build_bundles.sh
   PETGEM_FM_CSEM=build/fm.csem PETGEM_NP=2 pytest tests/e2e

   # Level 6 (make mms)
   bash tests/mms/build_bundle.sh 4
   PETGEM_MMS=build/mms PETGEM_NP=2 pytest tests/mms

   # MT half-space
   bash tests/mt/build_bundle.sh
   PETGEM_FM_MT=build/fm.mt pytest tests/mt

   # Reduced inversion
   bash tests/im/build_bundle.sh
   PETGEM_IM_CSEM=build/im.csem pytest tests/im

A bare ``pytest`` runs everything available. Tests whose toolchain, binary or
bundle is missing are skipped. Without the ``PETGEM_*`` variables the binaries
are looked up in ``build/``, the current directory and ``PATH``.
``PETGEM_NP`` sets the number of MPI ranks. ``pytest -m "not slow"`` runs
only the tests that do not start a kernel.

Tolerances
----------
- Levels 1-3 check analytic properties (DOF counts, reproduction of constant
  fields, partition of unity, symmetry, :math:`K_e G_e = 0`) to about
  ``1e-9``.
- Level 5 compares against the goldens with a relative tolerance of ``1e-3``.
- Level 6 uses ``atol + rtol*|golden|`` with ``rtol = 1e-3``, ``atol = 1e-8``,
  and requires :math:`\|Ax-b\|/\|b\| < 10^{-10}`.
- The MT half-space must give :math:`\rho = 100` Ω·m within 1 % and
  :math:`\phi_{xy} = 135^\circ`, :math:`\phi_{yx} = -45^\circ` within 0.5°.
- The reduced inversion matches its golden RMS history and model norm to
  ``rtol = 1e-8``.

Every test that solves a system uses a direct MUMPS solve, so none depends
on iterative convergence.

The goldens are regenerated only after an intended change of the results:

.. code-block:: bash

   bash examples/fm_csem_unit_cube/scripts/regen_golden.sh  # levels 4-5
   bash tests/mms/regen_golden.sh                           # level 6

Continuous integration
----------------------
``.github/workflows/ci-develop.yml`` builds the binaries and the MMS program
in the CI image and runs ``tests-fm-csem.yml`` (harnesses, levels 4-6),
``tests-fm-mt.yml`` (MT half-space) and ``tests-im-csem.yml`` (reduced
inversion). Documentation
jobs run only when the docs or sources change.
