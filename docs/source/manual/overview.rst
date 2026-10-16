========
Overview
========

**PETGEM** has four kernels built on the same Nédélec finite-element core:

- ``fm.csem`` - frequency-domain CSEM forward modeling (:doc:`fd_modeling`).
- ``fm.tdcsem`` - time-domain CSEM forward modeling (:doc:`td_modeling`).
- ``fm.mt`` - MT forward modeling (:doc:`mt_modeling`).
- ``im.csem`` - frequency-domain CSEM inversion (:doc:`inverse_modeling`).

Each kernel is a binary in ``build/``. The ``petgem`` dispatcher runs the same
code, selected by a subcommand or by ``-mode``:

.. code-block:: bash

   # Select the kernel with a subcommand
   petgem fm.csem -options_file params.txt

   # Select the kernel with -mode
   petgem -mode fm.csem -options_file params.txt

The kernel name is used unchanged everywhere: binaries, dispatcher
subcommands, the preprocessing ``-mode`` and the ``simulation_type`` attribute
of the output files.

Workflow
--------
Every run follows the same steps:

1. Build a tetrahedral mesh (see :doc:`meshing`).
2. Write the conductivity table ``sigmas.txt`` and the survey files.
3. Run ``utils/preprocess.py`` to produce the **input bundle** (HDF5) and a
   **parameter file** (PETSc options).
4. Run the kernel with ``-options_file``.
5. Read the output HDF5 (see :doc:`python_api`).

The bundle holds the data of the case: mesh, per-cell conductivity,
polynomial order and receivers, plus the sources (CSEM), the frequency (MT),
the waveform (time-domain CSEM) or the observed data (inversion). The
parameter file holds the solver and run options. Both formats are described in
:doc:`formats`.

Preprocessing
-------------
.. code-block:: bash

   # Build the input bundle and the parameter file
   python3 utils/preprocess.py -mode <kernel> -order <p> -case_dir <dir> \
       -mesh_filename <mesh> -receiver_filename <receivers> -sigma_file <sigmas> \
       [mode-specific inputs] [-input_filename input.h5] [-params_filename params.txt]

File names are relative to ``-case_dir``. Each mode needs its own inputs and
rejects those of the other modes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``-mode``
     - Required inputs
   * - ``fm.csem``
     - ``-source_filename``
   * - ``fm.tdcsem``
     - ``-source_filename`` (7 fields per row), ``-tdcsem_waveform_filename``,
       ``-tdcsem_t_end``
   * - ``fm.mt``
     - ``-mt_frequency_filename``
   * - ``im.csem``
     - ``-im_source_filename``, ``-observed_filename`` (and optionally
       ``-error_level``)

``-output_vtk model.vtu`` also writes the conductivity model for ParaView.
``python3 utils/preprocess.py -h`` lists every argument.

Common kernel options
---------------------
All kernels read their options from the PETSc options database
(``-options_file`` or the command line):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``-input_filename``
     - Input bundle. Required.
   * - ``-output_dir``
     - Output directory, created if absent. Required.
   * - ``-output_filename``
     - Output stem; the kernel writes ``<output_dir>/<stem>.h5``. Required.
   * - ``-order <1..6>``
     - Overrides the order stored in the bundle.

Any other option goes to PETSc (see :doc:`solver`). ``-help intro`` prints a
short usage summary, ``-help`` the full option list, and ``--version`` the
version.

Polynomial order
----------------
The order of the Nédélec basis is 1 to 6. The mesh does not depend on it.
``preprocess.py -order`` stores it in the bundle, and the kernel's ``-order``
overrides it, so one bundle can be run at any order.

Example cases
-------------
The cases in ``examples/`` share one layout: ``geometry/``, ``survey/``,
``configs/``, ``scripts/`` and, when the reference is a data file,
``reference/``. Generated files go to ``outputs/``, which is git-ignored.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Case
     - Kernel
     - Content
   * - ``examples/fm_csem_unit_cube``
     - ``fm.csem``
     - Homogeneous unit cube used by the test suite (:doc:`fd_examples`)
   * - ``examples/fm_csem_marine``
     - ``fm.csem``
     - Marine CSEM model with a thin resistive layer (:doc:`fd_examples`)
   * - ``examples/fm_tdcsem_halfspace``
     - ``fm.tdcsem``
     - Half-space time-domain CSEM with an analytical solution (:doc:`td_examples`)
   * - ``examples/fm_mt_hill``
     - ``fm.mt``
     - MT trapezoidal hill with two reference solutions (:doc:`mt_examples`)
   * - ``examples/im_csem_block``
     - ``im.csem``
     - Conductive block recovered from synthetic data
       (:doc:`inverse_examples`)
   * - ``examples/im_csem_checkerboard``
     - ``im.csem``
     - Two conductive and two resistive blocks (:doc:`inverse_examples`)