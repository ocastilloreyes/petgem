========
Overview
========

**PETGEM** provides two execution modes built on the same high-order vector
finite element core: **forward modeling** of the 3D Controlled-Source
Electromagnetic (CSEM) response, and **inverse modeling** (inversion) that
recovers a subsurface conductivity model from observed data. Both modes share
the same meshes, conductivity description, and preprocessing pipeline, and
differ only in the kernel that consumes the prepared input.

Execution modes
---------------
**PETGEM** ships three binaries (built by ``make`` into ``build/``):

- ``fm.csem``: the forward kernel (single-purpose).
- ``im.csem``: the inverse kernel (single-purpose).
- ``petgem``: a unified dispatcher that selects the mode from the command line:

  .. code-block:: bash

     ./petgem modeling -options_file params.txt   # forward  (== fm.csem)
     ./petgem inverse  -options_file params.txt   # inverse  (== im.csem)

The single-purpose binaries and the matching ``petgem`` mode run identical
code paths; use whichever fits your workflow.

Shared workflow
---------------
Regardless of mode, a **PETGEM** run follows the same high-level stages:

1. Generate or import a tetrahedral mesh using `Gmsh <http://gmsh.info/>`_.
2. Describe the subsurface conductivity model in a text table (``sigmas.txt``).
3. Run ``utils/preprocess.py`` to assemble a single **input bundle** and a
   matching **parameter file**.
4. Execute the appropriate kernel (``fm.csem`` or ``im.csem``).
5. Post-process the results.

Stages 1-3 are common to both modes; the mode-specific details live in
:doc:`forward_modeling` and :doc:`inverse_modeling`.

The input bundle
****************
``utils/preprocess.py`` is the centralized preprocessing entry point for both
modes. It reads the mesh, conductivity table, receivers, and sources, and
writes a **single unified HDF5 bundle** (``-input_filename``, default
``input.h5``) together with a PETSc-style parameter file (``-params_filename``,
default ``params.txt``). The kernel reads everything case-specific from the
bundle - the polynomial order (``/nord``), the model, the receivers, and (for
inverse runs) the multi-frequency sources, observed data, noise level, and
fixed-material list. The parameter file carries only solver and runtime knobs.

The mode is selected with ``-mode``:

.. code-block:: bash

   python3 utils/preprocess.py -mode forward ...   # forward bundle + params
   python3 utils/preprocess.py -mode inverse ...   # inverse bundle + params

Conductivity model
*******************
The conductivity model is a whitespace text table passed via ``-sigma_file`` (relative to
``-case_dir``). Each row is one material, indexed by 0-based material id
(``gmsh:physical - 1``):

.. code-block::

   # sigma_x sigma_y sigma_z [fixed]
   0.1 0.1 0.1 1     # held fixed during inversion (e.g. air, ocean)
   1.0 1.0 1.0 0     # invertable
   2.0 2.0 2.0       # 'fixed' column omitted -> defaults to 0

The optional fourth column (``fixed``) is only meaningful for inverse modeling:
a non-zero entry marks the material as held fixed during inversion. Forward
modeling ignores it.

Pre- and post-processing
************************
**PETGEM** provides a suite of Python helpers (under ``utils/`` and per-case
scripts) for:

- Mesh conversion into the bundle's expected layout.
- Conductivity model assembly from ``sigmas.txt``.
- Visualization of the model in VTK format (``-output_vtk``).
- Comparison of responses against a reference (per-case ``postprocess.py``).

Example cases ship under ``tests/cases/`` (e.g. ``tests/cases/csem_model`` for
forward modeling and ``tests/cases/inverse`` for inversion). These illustrate
the construction of the bundle and parameter files for each mode.
