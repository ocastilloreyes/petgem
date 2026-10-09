========
Overview
========

**PETGEM** provides three execution modes built on the same high-order Nédélec
finite-element core: **forward modeling** of the 3D CSEM response,
**inverse modeling**, which recovers a per-material conductivity model from
observed data, and **MT forward modeling** of the 3D magnetotelluric response.
All modes read the same kind of HDF5 input bundle, produced by the same
preprocessing script.

Execution modes
---------------
``make`` builds four binaries into ``build/``:

- ``fm.csem`` - CSEM forward kernel.
- ``im.csem`` - CSEM inverse kernel.
- ``fm.mt`` - MT forward kernel.
- ``petgem`` - dispatcher, selecting the mode from the command line:

  .. code-block:: bash

     ./petgem fm -options_file params.txt   # CSEM forward  (== fm.csem)
     ./petgem im -options_file params.txt   # CSEM inverse  (== im.csem)
     ./petgem mt -options_file params.txt   # MT forward    (== fm.mt)

The dispatcher calls the same ``runForward`` / ``runInverse`` /
``runMtForward`` entry points as the single-purpose binaries, so the code paths
are identical. ``-mode fm`` / ``-mode im`` / ``-mode mt`` is accepted as an
alternative to the positional subcommand, and ``forward``/``modeling`` and
``inverse`` are accepted as aliases of ``fm`` and ``im``.

Naming convention
*****************
``fm``, ``im`` and ``mt`` are the canonical tags for the three simulation
types, used consistently across the whole interface: the binaries
(``fm.csem``, ``im.csem``, ``fm.mt``), the dispatcher subcommands, the
preprocess ``-mode``, the kernel options (``-im_*`` for the inversion,
``-mt_*`` for MT), and the ``simulation_type`` attribute stamped into every
output file (``fm``, ``im``, ``fm.mt``).

Shared workflow
---------------
1. Generate or import a tetrahedral mesh (`Gmsh <http://gmsh.info/>`_ ``.msh``,
   or VTK ``.vtk``/``.vtu`` - see :doc:`meshing`).
2. Describe the conductivity model in a text table (``sigmas.txt``).
3. Run ``utils/preprocess.py`` to assemble the **input bundle** and a matching
   **parameter file**.
4. Run the kernel (``fm.csem``, ``im.csem`` or ``fm.mt``).
5. Post-process the results.

The input bundle
****************
``utils/preprocess.py`` is the entry point for all modes. It reads the mesh,
conductivity table, receivers, and sources or frequency, and writes a single HDF5 bundle
(``-input_filename``, default ``input.h5``) plus a PETSc options file
(``-params_filename``, default ``params.txt``).

The bundle carries everything case-specific: the mesh, the per-cell
conductivity, the polynomial order (``/order``), the receivers, the sources
(CSEM) or the frequency (MT), and - for inverse runs - the observed data, the
noise level, and the fixed-material list. The parameter file carries only solver and runtime
options. The layout is documented in :doc:`formats`.

The mode is selected with ``-mode``:

.. code-block:: bash

   python3 utils/preprocess.py -mode fm ...   # CSEM forward bundle + params
   python3 utils/preprocess.py -mode im ...   # CSEM inverse bundle + params
   python3 utils/preprocess.py -mode mt ...   # MT forward bundle + params

Forward mode requires ``-source_filename``; inverse mode requires
``-im_source_filename`` and ``-observed_filename``; MT mode requires
``-mt_frequency_filename``.

Polynomial order
****************
The Nédélec basis order is an integer in ``1..6``. It is passed to the
preprocess with ``-order`` and stored in the bundle as ``/order``. The kernels
read it from the bundle, and accept ``-order`` at run time as an override -
which also bypasses the stored value, so one bundle can serve every order.

Conductivity model
******************
The conductivity model is a whitespace text table passed via ``-sigma_file``
(relative to ``-case_dir``). Each row is one material, indexed by 0-based
material id:

.. code-block::

   # sigma_x sigma_y sigma_z [fixed]
   0.1 0.1 0.1 1     # held fixed during inversion (e.g. air, ocean)
   1.0 1.0 1.0 0     # invertable
   2.0 2.0 2.0       # 'fixed' column omitted -> defaults to 0

The optional fourth column (``fixed``) is only meaningful for inverse modeling:
a non-zero entry marks the material as held fixed. Forward modeling ignores it.

Pre- and post-processing
************************
The Python package under ``utils/`` (importable as ``petgem``) provides:

- Mesh and model reading, and assembly of the input bundle
  (``runPreprocessing``).
- Optional export of the conductivity model to VTU for visualization
  (``-output_vtk``).
- Readers for the bundle and for the kernel responses (``readBundle``,
  ``readResponses``, ``readAllResponses``) - see :doc:`python_api`.

Comparison against a reference is done by the per-case ``scripts/postprocess.py``.

Example cases
*************
The cases under ``examples/`` share the same layout (``geometry/``,
``survey/``, ``configs/``, ``scripts/``, ``reference/``), among them:

- ``examples/fm`` - a marine CSEM benchmark with a thin resistive
  layer, with a precomputed reference for validation.
- ``examples/unit_cube`` - a small homogeneous cube; the dataset the test suite
  is built around.
- ``examples/im1`` - the CSEM inversion benchmark: a buried conductive block
  recovered from noisy synthetic observations.
- ``examples/mt1`` - the MT trapezoidal hill of Castillo-Reyes et al. (2022),
  with two reference solutions.

The first two are forward cases (see :doc:`examples`); ``examples/im1`` covers
the full inverse workflow (see :doc:`inverse_examples`); ``examples/mt1`` the MT
workflow (see :doc:`mt_examples`).
