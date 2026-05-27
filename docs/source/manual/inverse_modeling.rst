================
Inverse modeling
================

The inverse kernel in **PETGEM** recovers a subsurface conductivity model from
observed CSEM data. It is built on the same high-order vector finite element
core as the forward kernel: each iteration solves forward and adjoint problems
on the unstructured tetrahedral mesh and assembles a gradient of the data
misfit with respect to the per-cell conductivity. The model is updated with a
limited-memory BFGS (L-BFGS) scheme, with Tikhonov regularization and an
optional neighbor-smoothing term to stabilize the recovered model.

Workflow
--------
The typical workflow for CSEM inversion in **PETGEM** mirrors the forward one
(the shared stages are described in :doc:`overview`), with inverse-specific
inputs:

1. Generate or import a mesh using `Gmsh <http://gmsh.info/>`_.
2. Define the starting/background conductivity model in ``sigmas.txt``,
   marking any materials to be held fixed with the ``fixed`` column.
3. Run ``utils/preprocess.py -mode inverse`` to assemble the input bundle. In
   addition to the model and receivers, the bundle embeds the multi-frequency
   sources, the observed data, the noise level, and the fixed-material list.
4. Execute the inverse kernel (``im.csem``) to recover the model.
5. Inspect the recovered model and convergence (optionally via VTU snapshots).

Inverse inputs
**************
Inverse preprocessing requires two inputs not used by forward modeling, both
embedded into the bundle:

- ``-inv_source_filename``: a **multi-frequency** sources text file. Unlike the
  forward source file (one frequency, many dipoles), each row carries 8 fields
  - ``freq x y z current length dip azimuth`` - one per (frequency, dipole)
  pair. Stored in the bundle under ``/sources/*``.
- ``-observed_filename``: the observed field, as either an HDF5 file with
  ``/Ex [N_freq, N_recv]`` (``complex128``, {r,i} compound) **or** a raw
  MATLAB-style ``invEx.dat`` text file (parsed inline - no separate conversion
  step). Stored in the bundle under ``/observed/Ex``. See :doc:`formats` for
  both layouts.

Two further inversion inputs are derived during preprocessing and embedded into
the bundle, each overridable on the kernel command line:

- **Noise level** (``/observed`` ``@error_level``): the relative data error used
  to weight the misfit. Set at preprocessing time with ``-error_level`` (or read
  from an HDF5 observed file's attribute); override at run time with
  ``-inv_error_level``.
- **Fixed materials** (``/inv_meta/fixed_materials``): the 0-based material ids
  flagged in the ``fixed`` column of ``sigmas.txt``; their gradient is zeroed
  and the smoother treats them as self-referencing. Override with
  ``-inv_fixed_materials``.

Parameter files
***************
The inverse parameter file (emitted by ``utils/preprocess.py``) carries the
bundle path plus solver and L-BFGS tuning knobs. Because the inverse kernel
re-solves the system at every iteration, the default solver is a direct
factorization (MUMPS). A representative inverse parameter file is:

.. code-block::

   -input_filename <case_dir>/input.h5
   -ksp_type preonly
   -pc_type                    lu
   -pc_factor_mat_solver_type  mumps
   -mat_mumps_icntl_14         80
   -mat_mumps_icntl_28         1
   -inv_max_iter               150
   -inv_lbfgs_memory           2
   -inv_lambda                 0.1
   -inv_diag_weight            0.0
   -inv_gtol                   1.0e-5
   -inv_rms_tol                1.05
   -inv_snapshot_interval      1
   -output_dir <case_dir>/
   -output_filename responses

The inversion knobs are:

- ``-inv_max_iter``: Maximum number of L-BFGS iterations.
- ``-inv_lbfgs_memory``: Number of correction pairs kept by L-BFGS.
- ``-inv_lambda``: Tikhonov regularization factor.
- ``-inv_diag_weight``: Self-weight in the neighbor smoother (``0`` disables it).
- ``-inv_gtol``: Gradient-norm convergence tolerance.
- ``-inv_rms_tol``: RMS misfit early-stop threshold.
- ``-inv_snapshot_interval``: Write a conductivity VTU snapshot every N accepted
  steps (``0`` disables snapshots; output goes to ``-output_dir``).

These may be set in the parameter file or overridden on the command line. The
data-derived overrides ``-inv_error_level`` and ``-inv_fixed_materials`` are
accepted but absent from the default template, since their values come from the
bundle.

Running PETGEM
**************
A typical command for a parallel inverse execution is:

.. code-block:: bash

   mpirun -n 4 build/im.csem -options_file path_to_params_file.txt

or, equivalently, through the unified dispatcher:

.. code-block:: bash

   mpirun -n 4 build/petgem inverse -options_file path_to_params_file.txt

A complete, runnable walkthrough is given in :doc:`examples_inverse`.
