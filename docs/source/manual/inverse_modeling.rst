================
Inverse modeling
================

``im.csem`` recovers a conductivity model from observed CSEM data. It reuses
the forward kernel's assembly and solver: each iteration solves forward and
adjoint problems for every frequency, assembles the gradient of the data misfit
by the adjoint-state method, and updates the model with L-BFGS. The model is
parameterized **per material**, not per cell: one conductivity value per row of
``sigmas.txt``, with materials flagged ``fixed`` excluded from the update.

Workflow
--------
1. Generate or import a mesh (see :doc:`meshing`).
2. Define the starting model in ``sigmas.txt``, marking materials to hold fixed
   with the ``fixed`` column.
3. Run ``utils/preprocess.py -mode im`` to assemble the input bundle.
4. Run ``im.csem``.
5. Inspect the recovered model (optionally via VTU snapshots).

Inverse inputs
--------------
Inverse preprocessing requires two inputs that forward modeling does not, both
embedded into the bundle:

- ``-im_source_filename`` - a **multi-frequency** source file: one row per
  ``(frequency, dipole)`` pair, 8 fields each
  (``freq x y z current length dip azimuth``). Stored under ``/sources``.
- ``-observed_filename`` - the observed field, either an HDF5 file with
  ``/Ex [N_freq, N_recv]`` or a raw MATLAB-style ``invEx.dat`` text file (parsed
  inline). Stored under ``/observed/Ex``. Both layouts are described in
  :doc:`formats`.

Two further quantities are derived during preprocessing and embedded in the
bundle, each overridable on the kernel command line:

- **Noise level** (``/observed`` ``@error_level``) - the relative data error
  used to weight the misfit. Set with ``-error_level`` at preprocessing time
  (or read from the HDF5 file's attribute); override at run time with
  ``-im_error_level``.
- **Fixed materials** (``/im_meta/fixed_materials``) - the 0-based material ids
  flagged in the ``fixed`` column of ``sigmas.txt``. Override with
  ``-im_fixed_materials``.

Kernel options
--------------
``im.csem`` accepts the forward kernel's required options (``-input_filename``,
``-output_dir``, ``-output_filename``) and ``-order``, plus the inversion
options below. Values in brackets are the kernel defaults, which apply when the
option is absent from both the parameter file and the command line.

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Option
     - Default
     - Meaning
   * - ``-im_max_iter``
     - ``80``
     - Maximum L-BFGS iterations.
   * - ``-im_lbfgs_memory``
     - ``5``
     - Number of correction pairs kept by L-BFGS.
   * - ``-im_lambda``
     - ``0.1``
     - Tikhonov regularization weight.
   * - ``-im_error_level``
     - ``0.01``
     - Relative data-error level. Overrides the bundle's ``@error_level``.
   * - ``-im_gtol``
     - ``1e-5``
     - Gradient-norm convergence tolerance.
   * - ``-im_rms_tol``
     - ``0`` (off)
     - Absolute RMS misfit early-stop threshold.
   * - ``-im_rms_rtol``
     - ``1e-3``
     - Relative RMS-improvement threshold for the plateau stop. ``0`` disables it.
   * - ``-im_rms_stall_window``
     - ``3``
     - Consecutive iterations below ``-im_rms_rtol`` required to declare a
       plateau and stop.
   * - ``-im_diag_weight``
     - ``0`` (off)
     - Self-weight of the neighbor smoother applied to the gradient.
   * - ``-im_fixed_materials``
     - from bundle
     - Comma-separated material ids excluded from the update and from smoothing.
   * - ``-im_snapshot_interval``
     - ``0`` (off)
     - Write a VTU snapshot every N accepted L-BFGS steps, into ``-output_dir``.
       Each snapshot carries a single cell field, ``rho`` (Ohm.m).
   * - ``-im_observed_mode``
     - ``external``
     - Where the observed data comes from. ``external`` reads the bundle's
       ``/observed/Ex``; ``fm_native`` reads an ``fm.csem`` responses file
       through its ``/sources/src{k}/fields/Ex`` layout (source ``k`` →
       frequency row ``k``).
   * - ``-im_observed_file``
     - bundle
     - Path to the observed-data file. Optional in ``external`` mode (defaults
       to ``-input_filename``); **required** in ``fm_native`` mode. Any noise
       must already be present in that file.

Besides the absolute ``-im_rms_tol``, the L-BFGS loop stops when the RMS misfit
plateaus: when the relative improvement stays below ``-im_rms_rtol`` for
``-im_rms_stall_window`` consecutive iterations.

Parameter file
**************
``utils/preprocess.py -mode im`` emits:

.. code-block::

   -input_filename <case_dir>/input.h5
   -ksp_type preonly
   -pc_type                    lu
   -pc_factor_mat_solver_type  mumps
   -mat_mumps_icntl_14         80
   -mat_mumps_icntl_28         1
   -im_max_iter               150
   -im_lbfgs_memory           2
   -im_lambda                 0.1
   -im_diag_weight            0.0
   -im_gtol                   1.0e-5
   -im_rms_tol                1.05
   -im_snapshot_interval      1
   -output_dir <case_dir>/
   -output_filename responses

This template overrides several kernel defaults and selects a **direct** solve
(MUMPS) instead of the iterative forward default; see :doc:`solver`.

``-im_error_level`` and ``-im_fixed_materials`` are accepted but absent from
the template, since their values come from the bundle.

Running
-------
.. code-block:: bash

   mpirun -n 4 build/im.csem -options_file path/to/params.txt

or through the dispatcher:

.. code-block:: bash

   mpirun -n 4 build/petgem im -options_file path/to/params.txt

Any option can be overridden on the command line, e.g. ``-im_max_iter 30``.
