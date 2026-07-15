========
Overview
========

**PETGEM** provides two execution modes built on the same high-order Nédélec
finite-element core: **forward modeling** of the 3D CSEM response, and
**inverse modeling**, which recovers a per-material conductivity model from
observed data. Both modes read the same kind of HDF5 input bundle, produced by
the same preprocessing script.

Execution modes
---------------
``make`` builds three binaries into ``build/``:

- ``fm.csem`` - forward kernel.
- ``im.csem`` - inverse kernel.
- ``petgem`` - dispatcher, selecting the mode from the command line:

  .. code-block:: bash

     ./petgem fm -options_file params.txt   # forward  (== fm.csem)
     ./petgem im -options_file params.txt   # inverse  (== im.csem)

The dispatcher calls the same ``runForward`` / ``runInverse`` entry points as
the single-purpose binaries, so the code paths are identical. ``-mode fm`` /
``-mode im`` is accepted as an alternative to the positional subcommand, and
``forward``/``modeling`` and ``inverse`` are accepted as aliases of ``fm`` and
``im``.

Naming convention
*****************
``fm`` and ``im`` are the canonical tags for the two simulation types, used
consistently across the whole interface: the binaries (``fm.csem``,
``im.csem``), the dispatcher subcommands, the preprocess ``-mode``, the
inversion options (every one is ``-im_*``), and the ``simulation_type``
attribute stamped into every output file.

Shared workflow
---------------
1. Generate or import a tetrahedral mesh (`Gmsh <http://gmsh.info/>`_ ``.msh``,
   or VTK ``.vtk``/``.vtu`` - see :doc:`meshing`).
2. Describe the conductivity model in a text table (``sigmas.txt``).
3. Run ``utils/preprocess.py`` to assemble the **input bundle** and a matching
   **parameter file**.
4. Run the kernel (``fm.csem`` or ``im.csem``).
5. Post-process the results.

The input bundle
****************
``utils/preprocess.py`` is the entry point for both modes. It reads the mesh,
conductivity table, receivers, and sources, and writes a single HDF5 bundle
(``-input_filename``, default ``input.h5``) plus a PETSc options file
(``-params_filename``, default ``params.txt``).

The bundle carries everything case-specific: the mesh, the per-cell
conductivity, the polynomial order (``/order``), the receivers, the sources,
and - for inverse runs - the observed data, the noise level, and the
fixed-material list. The parameter file carries only solver and runtime
options. The layout is documented in :doc:`formats`.

The mode is selected with ``-mode``:

.. code-block:: bash

   python3 utils/preprocess.py -mode fm ...   # forward bundle + params
   python3 utils/preprocess.py -mode im ...   # inverse bundle + params

Forward mode requires ``-source_filename``; inverse mode requires
``-im_source_filename`` and ``-observed_filename``.

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

Comparison against a reference is done by the per-case ``postprocess.py``.

Example cases
*************
Two cases ship under ``examples/``:

- ``examples/canonical_model`` - a marine CSEM benchmark with a thin resistive
  layer, with a precomputed reference for validation.
- ``examples/unit_cube`` - a small homogeneous cube; the dataset the test suite
  is built around.

Both are forward cases. See :doc:`examples`.
