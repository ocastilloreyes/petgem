=========
Changelog
=========

Notable changes to **PETGEM** are recorded here. Versions follow the
``include/version.h`` header, which is the single source of truth for the
release string.

2.0.0
-----
Major refactor of **PETGEM** to C and PETSc.

- Refactored the solver in C and integrated with `PETSc <https://petsc.org/release/>`_
  for scalability on massively parallel architectures, memory efficiency, and
  mesh handling.
- High-order Nédélec (edge) vector finite elements, orders 1 to 6, on
  unstructured tetrahedral meshes.
- **Forward modeling** (``fm.csem``) and **inverse modeling** (``im.csem``,
  L-BFGS with adjoint-state gradients), plus a unified ``petgem`` dispatcher
  (``petgem modeling`` / ``petgem inverse``).
- Unified HDF5 input bundle assembled by ``utils/preprocess.py`` for both modes
  (see :doc:`formats`).
- BDDC-preconditioned iterative solve for forward modeling; direct (MUMPS)
  solve for inversion and for high polynomial order (see :doc:`solver`).
- Optional `Extrae <https://tools.bsc.es/extrae>`_ instrumentation
  (``make USE_EXTRAE=1``) for performance profiling.

Earlier history
---------------
**PETGEM** was originally developed in Python; for the history of those
releases and the underlying methods, see the publications in
:doc:`publications`.
