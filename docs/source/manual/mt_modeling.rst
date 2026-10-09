=============
MT modeling
=============

``fm.mt`` computes the 3D magnetotelluric (MT) response of a conductivity model
at a single frequency, discretizing the total electric field with Nédélec
elements of order 1 to 6 on an unstructured tetrahedral mesh. It shares the
mesh loading, the Nédélec space, the operator assembly, the solver, and the
receiver interpolation with ``fm.csem``; only the boundary condition, the
right-hand side, and the responses are MT-specific. The formulation is
described in :doc:`method`.

Workflow
--------
1. Generate or import a mesh of an axis-aligned box (see :doc:`meshing`).
2. Define the conductivity model in ``sigmas.txt``.
3. Run ``utils/preprocess.py -mode mt`` to produce the input bundle and the
   parameter file.
4. Run ``fm.mt``.
5. Post-process the responses.

Domain requirements
-------------------
- The domain must be an **axis-aligned box**. Every boundary face is assigned
  to one of the six box faces (top, bottom, and four lateral faces); the kernel
  stops if a boundary face is not on the bounding box.
- The conductivity on the **lateral faces** must vary only with ``z`` and be
  horizontally isotropic (``sigma_x = sigma_y``): the kernel builds the 1D
  profile of the boundary field from them and stops if two lateral faces
  disagree at the same height.
- The boundaries must be far enough from the survey for the 1D boundary field
  to be accurate. Castillo-Reyes et al. (2022) recommend at least four skin
  depths; see :doc:`mt_examples`.

Kernel options
--------------
``fm.mt`` takes its options from a PETSc options file
(``-options_file params.txt``) or directly on the command line.

**Required**

- ``-input_filename`` - the input bundle (HDF5).
- ``-output_dir`` - output directory (created if absent).
- ``-output_filename`` - output stem; writes ``<output_dir>/<stem>.h5``.

**Optional**

- ``-order <1..6>`` - override the bundle's ``/order``.
- ``-mt_1d_equation <paper|h>`` - equation of the 1D boundary field.
  ``paper`` (default) is Eq. (11) of Castillo-Reyes et al. (2022),
  :math:`H'' + i\omega\mu\sigma H = 0`; ``h`` is
  :math:`(\rho H')' + i\omega\mu H = 0`. See :doc:`method`.
- ``-mt_1d_refine <n>`` - 1D element size is the smallest 3D edge length on the
  lateral faces of each layer divided by ``n``. Default: 10.

Everything else in the options file is passed to PETSc; see :doc:`solver`.

``-help intro`` prints a usage summary and exits; ``-help`` prints the full
PETSc option database; ``--version`` prints the version.

Parameter file
**************
The parameter file emitted by ``utils/preprocess.py -mode mt`` uses a direct
solver:

.. code-block::

   -input_filename <case_dir>/input.h5
   -dm_mat_type aij
   -ksp_type preonly
   -pc_type lu
   -pc_factor_mat_solver_type mumps
   -output_dir <case_dir>/
   -output_filename responses

The polynomial order is **not** written here - the kernel reads it from the
bundle's ``/order`` dataset.

Input files
-----------

Frequency file
**************
``-mt_frequency_filename`` holds the frequency (Hz), one value; ``#`` comments
are allowed:

.. code-block::

   # Frequency (Hz)
   2.0

Each run solves the two source polarizations (x and y) at this frequency.

Receiver file
*************
``-receiver_filename`` lists receiver positions, one Cartesian point ``x y z``
per row, as for ``fm.csem``.

Running
-------
.. code-block:: bash

   mpirun -n 4 build/fm.mt -options_file path/to/params.txt

or through the dispatcher:

.. code-block:: bash

   mpirun -n 4 build/petgem mt -options_file path/to/params.txt

Output
------
The kernel writes a single HDF5 file, ``<output_dir>/<output_filename>.h5``,
with the fields of both polarizations, the impedance tensor, the apparent
resistivities, the phases, and the tipper at the receivers. The layout is
documented in :doc:`formats`.

A runnable walkthrough is given in :doc:`mt_examples`.
