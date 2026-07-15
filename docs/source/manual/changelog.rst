=========
Changelog
=========

Versions follow ``include/version.h``, the single source of truth for the
release string.

Unreleased
----------
Unification of the ``fm.csem`` and ``im.csem`` interfaces. The computational
pipelines, algorithms, and solver logic are unchanged.

- **Inversion options renamed** ``-inv_*`` → ``-im_*``, matching the kernel, the
  ``imParams`` struct, and the ``petgem im`` subcommand. The old spelling is
  retired and rejected with an error naming its replacement (see
  :doc:`troubleshooting`).
- ``-im_rms_rtol`` and ``-im_rms_stall_window`` are now parsed with the other
  inversion options, so they appear under ``-help`` and can be set from the
  parameter file. Their values and defaults are unchanged.
- **Unified output provenance.** Both kernels now write the same root attribute
  block through one shared routine - ``petgem_version``, ``simulation_type``,
  ``input_filename``, ``order``, ``ksp_type``, ``pc_type``, ``mpi_tasks``,
  ``date`` - so an inversion result can be traced to its input bundle, mesh,
  solver configuration, and rank count. Previously the inverse kernel wrote a
  differently-cased subset (``Nord``, ``Petgem_version``, …) and recorded
  neither its input file nor its timestamp. See :doc:`formats`.
- Inversion VTU snapshots are named from ``-output_filename`` like every other
  product, instead of a fixed ``inv_model_`` prefix.
- ``fm`` and ``im`` are the canonical simulation tags everywhere:
  ``preprocess.py -mode fm|im`` (with ``forward``/``inverse`` still accepted, as
  in the dispatcher), the bundle group ``/im_meta``, and the
  ``-im_source_filename`` preprocess argument.
- ``-help`` now groups the options shared by both kernels under
  ``PETGEM: required/optional options``. They were previously presented under an
  ``fm.csem:`` heading even when running ``im.csem``, which registers them
  through the same reader.
- ``petgem --help`` works. It previously aborted with an MPI error, because the
  dispatcher prints its usage before ``PetscInitialize``.

2.0.0
-----
Rewrite of **PETGEM** in C on top of PETSc. Relative to the earlier Python
implementation:

- Kernels written in C and built against `PETSc <https://petsc.org/release/>`_
  and MPI.
- High-order Nédélec (edge) elements, orders 1 to 6, on unstructured
  tetrahedral meshes.
- **Forward modeling** (``fm.csem``) and **inverse modeling** (``im.csem``,
  L-BFGS with adjoint-state gradients), plus the ``petgem`` dispatcher
  (``petgem modeling`` / ``petgem inverse``).
- A single HDF5 input bundle assembled by ``utils/preprocess.py`` for both
  modes (see :doc:`formats`).
- Mesh input from Gmsh ``.msh`` or VTK ``.vtk``/``.vtu`` (tetrahedral), with the
  format auto-detected (see :doc:`meshing`).
- BDDC-preconditioned iterative solve for forward modeling, configured with the
  discrete-gradient matrix assembled at the run's polynomial order; direct
  (MUMPS) solve for inversion (see :doc:`solver`).
- Method-of-manufactured-solutions verification mode (``fm.csem -mms``) and a
  six-level forward test suite (see :doc:`testing`).
- Optional `Extrae <https://tools.bsc.es/extrae>`_ instrumentation
  (``make USE_EXTRAE=1``).

Earlier history
---------------
For the original Python implementation and the underlying methods, see the
publications in :doc:`publications`.
