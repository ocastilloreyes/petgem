=======
Testing
=======

**PETGEM** ships a `pytest <https://docs.pytest.org/>`_ suite under ``tests/``,
split into fast unit tests and end-to-end integration tests.

Layout
------
- ``tests/unit/`` - fast, dependency-light tests of the Python helpers
  (bundle/sigma/source readers). No compiled kernel required.
- ``tests/integration/`` - end-to-end pipeline tests
  (preprocess :math:`\rightarrow` ``fm.csem`` :math:`\rightarrow` postprocess),
  including a structural smoke test and a numerical regression against the
  ModEM reference, parametrized over ``nord``.
- ``examples/`` - the case data exercised by the integration tests.

Markers (declared in ``pytest.ini``):

- ``integration`` - end-to-end tests; **skipped automatically** when the
  ``fm.csem`` binary is unavailable, so the suite still passes on machines that
  have not built the C kernels.
- ``slow`` - tests that take more than a few seconds.

Running the tests
-----------------
.. code-block:: bash

   # Everything (unit always runs; integration skips without a binary)
   pytest

   # Unit tests only
   pytest tests/unit

   # Integration suite, streaming kernel output
   pytest tests/integration -m integration -ra -s

Environment variables
---------------------
- ``PETGEM_FM_CSEM`` - path to the ``fm.csem`` binary used by the integration
  tests. If unset, the suite searches ``<repo>/fm.csem``, ``<repo>/build/fm.csem``,
  then ``PATH``; if none exists, the integration tests skip.
- ``PETGEM_TEST_MPI_NPROC`` - number of MPI ranks for the kernel runs
  (default ``2``). Raise it on a many-core machine; the numerical result is
  invariant to the rank count (see :doc:`solver`).

.. code-block:: bash

   PETGEM_FM_CSEM=$PWD/build/fm.csem PETGEM_TEST_MPI_NPROC=4 \
       pytest tests/integration -m integration -ra -s

What the integration suite checks
---------------------------------
- **Smoke** (``test_fm_csem_smoke.py``): the kernel runs at ``nord`` 1, 2, 3 and
  produces finite field components sized to the receiver vector.
- **Reference** (``test_fm_csem_csem_model_reference.py``): the magnitude of
  ``Ex`` agrees with the shipped ModEM reference within the NRMSD tolerance
  (default ``0.03``), at ``nord`` 1, 2, 3.

The meshes are generated on demand from the per-order ``.geo`` files (the
``.msh`` files are not committed); a missing ``gmsh`` causes the affected test
to skip rather than fail.
